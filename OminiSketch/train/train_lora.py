"""Latest masked-source skelora training: setup, optimization, validation and saving.

Run: python -B OminiSketch/train/train_lora.py --name new_trial --samples 128
The companion helpers.py owns preprocessing, losses and split handling.
"""
import argparse
import inspect
import json
import math
import os
import random
import sys
from pathlib import Path
import yaml
import torch
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from diffusers import FluxPipeline
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.before import config as local_config, CONDITION_PROTOCOL
from core.processing import ATTENTION_PROTOCOL, packed_hole_mask, transformer_forward
from train.helpers import TrainingPreprocessor, SketchLoRADataset, load_config, make_image_ids, area_normalized_loss, unpack_visual_tokens

class VisualTrainer(TrainingPreprocessor):
    """Own model setup, training steps, validation and checkpoint selection."""

    def __init__(self, cfg, max_samples=None):
        self.kind = cfg.data.condition_image_kind
        self.empty_text = None
        self.vision_teacher = None
        self.loss_step = 0
        self.cfg = cfg
        self.max_samples = max_samples
        self.attention_scale = cfg.training.get('condition_attention_scale', 0.01)
        if not math.isfinite(self.attention_scale) or not 0 < self.attention_scale <= 1:
            raise ValueError('Training requires 0 < condition_attention_scale <= 1')
        if cfg.data.image_size != 512:
            raise ValueError('This trainer currently requires image_size=512')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            major, _ = torch.cuda.get_device_capability(self.device)
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        if self.device.type == 'cpu':
            print('[训练] ⚠ 未检测到 CUDA，将在 CPU 上训练 (极慢，仅用于调试)')
        print(f'[训练] 加载 FLUX 基座: {cfg.model.flux_base}')
        self.pipe = FluxPipeline.from_pretrained(cfg.model.flux_base, torch_dtype=self.dtype, local_files_only=True)
        for module in (self.pipe.vae, self.pipe.text_encoder, self.pipe.text_encoder_2):
            module.requires_grad_(False)
            module.eval()
        print(f'[训练] 配置 LoRA (rank={cfg.lora.rank}, alpha={cfg.lora.alpha})')
        self.lora_config = LoraConfig(r=cfg.lora.rank, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout, target_modules=cfg.lora.target_modules, bias='none')
        self.transformer = get_peft_model(self.pipe.transformer, self.lora_config, adapter_name='skelora')
        self.transformer.enable_gradient_checkpointing()
        if cfg.model.get('initial_lora'):
            from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
            initial_dir = Path(cfg.model.initial_lora)
            if (initial_dir / 'skelora').is_dir():
                initial_dir = initial_dir / 'skelora'
            set_peft_model_state_dict(self.transformer, load_peft_weights(str(initial_dir), device='cpu'), adapter_name='skelora')
            print(f'[warm-start] {initial_dir}')
        self.lora_params = [p for p in self.transformer.parameters() if p.requires_grad]
        print(f'[训练] 可训练 LoRA 参数: {len(self.lora_params)}')
        if cfg.training.get('perceptual_weight', 0) > 0:
            from transformers import SiglipVisionModel
            self.vision_teacher = SiglipVisionModel.from_pretrained(local_config.models.redux_image_encoder, torch_dtype=self.dtype, local_files_only=True).requires_grad_(False).eval().to(self.device)
            if self.vision_teacher.config.attention_dropout != 0:
                raise ValueError('Perceptual teacher must have zero attention dropout')
            self.vision_teacher.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            self.pipe.vae.enable_gradient_checkpointing()

    def train_step(self, cached):
        device, dtype = (self.device, self.dtype)
        x0 = cached['x0_tokens'].to(device, dtype)
        source = cached['x0_src_tokens'].to(device, dtype)
        mask = packed_hole_mask(cached['mask_weight_map'].to(device), 64, 64, 16, self.pipe._pack_latents).to(dtype)
        noise = torch.randn_like(x0)
        index = torch.randint(0, len(self.pipe.scheduler.timesteps), (1,)).long()
        sigma = self.pipe.scheduler.sigmas[index].to(device, dtype)
        timestep = self.pipe.scheduler.timesteps[index].to(device)
        clean = mask * x0 + (1 - mask) * source
        noisy = (1 - sigma) * clean + sigma * noise
        velocity = noise - clean
        drop = self.transformer.training and float(torch.rand(())) < self.cfg.training.text_dropout
        if drop:
            embeds, pooled, ids = self.empty_text
        else:
            embeds, pooled, ids = (cached['prompt_embeds'], cached['pooled'], cached['text_ids'])
        embeds, pooled, ids = (embeds.to(device, dtype), pooled.to(device, dtype), ids.to(device, dtype))
        if ids.ndim == 3:
            ids = ids[0]
        conditions = [cached['sketch_tokens'].to(device, dtype)]
        parts = [self._make_condition(tokens, 'skelora', 12 if i == 0 else 4) for i, tokens in enumerate(conditions)]
        latents, condition_ids, types, names = map(list, zip(*parts))
        guidance = torch.full((1,), self.cfg.training.guidance, device=device, dtype=dtype) if self.transformer.config.guidance_embeds else None
        with torch.amp.autocast(device.type, dtype=dtype):
            pred = transformer_forward(self.transformer, model_config={'condition_attention_scale': self.attention_scale, 'latent_lora': self.cfg.training.image_lora}, condition_latents=latents, condition_ids=condition_ids, condition_type_ids=types, condition_names=names, hidden_states=noisy, timestep=timestep.float() / 1000, guidance=guidance, pooled_projections=pooled, encoder_hidden_states=embeds, txt_ids=ids, img_ids=make_image_ids(noisy.shape[1], device, dtype), return_dict=False, use_cache=False)[0]
        loss = area_normalized_loss(pred, velocity, mask, self.cfg.training.background_loss_weight)
        if self.cfg.training.reconstruction_weight:
            estimate = noisy.float() - sigma.float() * pred.float()
            loss = loss + self.cfg.training.reconstruction_weight * ((estimate - clean.float()).abs() * mask.float()).sum() / mask.float().sum().clamp_min(1)
        weight = self.cfg.training.get('perceptual_weight', 0)
        every = self.cfg.training.get('perceptual_every', 4)
        training = self.transformer.training
        use_perceptual = self.vision_teacher is not None and (not training or self.loss_step % every == 0)
        if training:
            self.loss_step += 1
        if use_perceptual and float(sigma) < 0.8:
            self.pipe.vae.to(device)
            self.vision_teacher.train(training)
            self.pipe.vae.train(training)
            estimate = noisy.float() - sigma.float() * pred.float()
            estimate = mask.float() * estimate + (1 - mask.float()) * source.float()
            latent = unpack_visual_tokens(estimate.to(dtype))
            latent = latent / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
            pixels = self.pipe.vae.decode(latent, return_dict=False)[0]
            features = self._visual_features(pixels, cached['region_bbox'])
            target_features = cached['target_visual_feature'].to(device)
            semantic = (1 - (features * target_features).sum(dim=-1)).mean()
            loss = loss + weight * (every if training else 1) * semantic
        return loss

    @torch.no_grad()
    def _fixed_probe_loss(self, cache, limit=4):
        was_training = self.transformer.training
        self.transformer.eval()
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        try:
            with torch.random.fork_rng(devices=devices):
                losses = []
                for index, cached in enumerate(sorted(cache, key=lambda item: item['filename'])[:limit]):
                    torch.manual_seed(self.cfg.training.seed + 100000 + index)
                    losses.append(float(self.train_step(cached)))
            return sum(losses) / len(losses)
        finally:
            self.transformer.train(was_training)

    def _save_lora(self, path, adapter_name):
        if not Path(path).resolve().is_relative_to(Path(local_config.models.sketch_lora).resolve()):
            raise ValueError('Adapter writes must remain inside Project/model/skelora')
        sig = inspect.signature(self.transformer.save_pretrained)
        if 'selected_adapters' in sig.parameters:
            self.transformer.save_pretrained(path, selected_adapters=[adapter_name])
        elif 'adapter_name' in sig.parameters:
            self.transformer.save_pretrained(path, adapter_name=adapter_name)
        else:
            self.transformer.save_pretrained(path)
        adapter_dir = Path(path) / adapter_name
        if not (adapter_dir / 'adapter_config.json').is_file():
            adapter_dir = Path(path)
        metadata = {'condition_protocol': CONDITION_PROTOCOL, 'flux_base': str(self.cfg.model.flux_base), 'image_size': self.cfg.data.image_size, 'seed': self.cfg.training.seed, 'mask_source': 'dataset_mask_png', 'mask_edit_color': self.cfg.data.get('mask_edit_color', 'black'), 'attention_protocol': ATTENTION_PROTOCOL, 'condition_attention_scale': self.attention_scale, 'image_lora': self.cfg.training.get('image_lora', False), 'use_redux': self.cfg.training.get('use_redux', False), 'background_lock': self.cfg.training.get('background_lock', False), 'mask_packing': 'per_latent_pixel_area_v1', 'warm_start': bool(self.cfg.model.get('initial_lora')), 'timestep_source': 'scheduler.timesteps[index]'}
        (adapter_dir / 'training_protocol.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        folder = Path(path) / adapter_name
        if not (folder / 'training_protocol.json').exists():
            folder = Path(path)
        metadata = json.loads((folder / 'training_protocol.json').read_text())
        metadata.update(condition_image_kind=self.kind, loss_protocol='edit_area_normalized_mse_v1', text_dropout=self.cfg.training.text_dropout, background_loss_weight=self.cfg.training.background_loss_weight, reconstruction_weight=self.cfg.training.reconstruction_weight, perceptual_weight=self.cfg.training.get('perceptual_weight', 0), perceptual_every=self.cfg.training.get('perceptual_every', 4), perceptual_grid=self.cfg.training.get('perceptual_grid', 0), text_lora=self.cfg.training.get('text_lora', False), perceptual_teacher=('frozen_SigLIP_ROI_grid' if self.cfg.training.get('perceptual_grid', 0) else 'frozen_SigLIP_ROI_pooler') if self.vision_teacher is not None else None, source_reference_condition=False)
        (folder / 'training_protocol.json').write_text(json.dumps(metadata, indent=2))

    def train(self):
        t_cfg = self.cfg.training
        grad_accum = t_cfg.gradient_accumulation_steps
        num_epochs = t_cfg.num_epochs
        lr = t_cfg.learning_rate
        save_every = t_cfg.save_every_epochs
        dataset = SketchLoRADataset(self.cfg.data.train_dir, self.cfg.data.image_size, self.cfg.data.default_prompt, self.cfg.data.get('mask_edit_color', 'black'))
        validation_dataset = None
        if self.cfg.data.get('split_manifest'):
            manifest = json.loads(Path(self.cfg.data.split_manifest).read_text(encoding='utf-8'))
            partitions = validate_manifest(manifest, dataset.sample_dirs, dataset.data_dir)
            validation_dataset = SketchLoRADataset(self.cfg.data.train_dir, self.cfg.data.image_size, self.cfg.data.default_prompt, self.cfg.data.get('mask_edit_color', 'black'))
            dataset.sample_dirs = partitions['train']
            validation_dataset.sample_dirs = partitions['validation']
            random.Random(t_cfg.seed).shuffle(validation_dataset.sample_dirs)
            validation_dataset.sample_dirs = validation_dataset.sample_dirs[:t_cfg.get('validation_samples', 64)]
            print(f'[split] train={len(partitions['train'])}, validation={len(partitions['validation'])}, test={len(partitions['test'])}; fixed held-out validation subset={len(validation_dataset)}')
        if self.max_samples:
            random.Random(t_cfg.seed).shuffle(dataset.sample_dirs)
            dataset.sample_dirs = dataset.sample_dirs[:self.max_samples]
            print(f'[训练] 仅使用前 {self.max_samples} 个样本 (冒烟测试)')
        assert t_cfg.batch_size == 1, '当前按单样本前向实现，batch_size 固定为 1'
        if self.device.type == 'cuda':
            self.pipe.vae.to(self.device)
            self.pipe.text_encoder.to(self.device)
            self.pipe.text_encoder_2.to(self.device)
        print(f'[训练] 预编码 {len(dataset)} 个样本 (VAE + 文本编码器, 仅此一遍)...')
        cache = []
        for i in tqdm(range(len(dataset)), desc='预编码'):
            cache.append(self._preencode(dataset[i]))
        validation_cache = []
        if validation_dataset is not None:
            for i in tqdm(range(len(validation_dataset)), desc='验证预编码'):
                validation_cache.append(self._preencode(validation_dataset[i]))
        cache_bytes = sum((value.numel() * value.element_size() for item in cache + validation_cache for value in item.values() if isinstance(value, torch.Tensor)))
        print(f'[cache] CPU tensor cache={cache_bytes / 1024 ** 3:.2f} GiB')
        for module in (self.pipe.vae, self.pipe.text_encoder, self.pipe.text_encoder_2):
            module.to('cpu')
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.transformer.to(self.device)
        print(f'[训练] 预编码完成, 缓存 {len(cache)} 个样本, transformer 已上 GPU')
        optimizer = torch.optim.AdamW(self.lora_params, lr=lr, weight_decay=t_cfg.weight_decay)
        total_steps = math.ceil(len(cache) / grad_accum) * num_epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
        output_dir = self.cfg.model.lora_output_dir
        os.makedirs(output_dir, exist_ok=True)
        initial_probe = self._fixed_probe_loss(cache)
        if not math.isfinite(initial_probe):
            raise FloatingPointError('Initial fixed probe loss is non-finite')
        print(f'[probe] Initial fixed training-sample loss={initial_probe:.6f}')
        self._save_lora(os.path.join(output_dir, 'initial'), 'skelora')
        self._save_lora(output_dir, 'skelora')
        initial_validation = self._fixed_probe_loss(validation_cache, len(validation_cache)) if validation_cache else None
        if initial_validation is not None and (not math.isfinite(initial_validation)):
            raise FloatingPointError('Initial validation loss is non-finite')
        history = {'initial_fixed_probe_loss': initial_probe, 'probe_is_training_subset': True, 'initial_validation_loss': initial_validation, 'selection_metric': 'held_out_validation_loss' if validation_cache else 'training_probe_loss', 'validation_sample_ids': [item['filename'] for item in validation_cache], 'condition_attention_scale': self.attention_scale, 'epochs': []}
        (Path(output_dir) / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f'[训练] 开始 LoRA 训练: {num_epochs} epoch, 样本数={len(cache)}, 梯度累积={grad_accum}, lr={lr}')
        best_loss = initial_validation if validation_cache else initial_probe
        global_step = 0
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(num_epochs):
            self.transformer.train()
            epoch_loss = 0.0
            num_batches = 0
            random.Random(t_cfg.seed + epoch).shuffle(cache)
            pbar = tqdm(cache, desc=f'Epoch {epoch + 1}/{num_epochs}')
            grad_norms = []
            for batch_index, cached in enumerate(pbar):
                group_size = min(grad_accum, len(cache) - batch_index // grad_accum * grad_accum)
                raw_loss = self.train_step(cached)
                loss = raw_loss / group_size
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss; weights were not saved')
                loss.backward()
                global_step += 1
                if (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(cache):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.lora_params, t_cfg.max_grad_norm, error_if_nonfinite=True)
                    grad_norms.append(float(grad_norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                epoch_loss += raw_loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{raw_loss.item():.4f}'})
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f'\nEpoch {epoch + 1}: Loss={avg_loss:.4f}')
            if (epoch + 1) % save_every == 0:
                save_path = os.path.join(output_dir, f'sketch_lora_epoch{epoch + 1}')
                self._save_lora(save_path, 'skelora')
                print(f'[保存] {save_path}')
            record = {'epoch': epoch + 1, 'random_training_loss': avg_loss, 'mean_gradient_norm': sum(grad_norms) / len(grad_norms)}
            probe_every = t_cfg.get('probe_every_epochs', 5)
            if (epoch + 1) % probe_every == 0 or epoch + 1 == num_epochs:
                probe_loss = self._fixed_probe_loss(cache)
                record['fixed_probe_loss'] = probe_loss
                print(f'[probe] Epoch {epoch + 1}: fixed loss={probe_loss:.6f}')
                if not math.isfinite(probe_loss):
                    raise FloatingPointError('Fixed probe loss is non-finite')
                selection_loss = probe_loss
                if validation_cache:
                    selection_loss = self._fixed_probe_loss(validation_cache, len(validation_cache))
                    record['held_out_validation_loss'] = selection_loss
                    print(f'[validation] Epoch {epoch + 1}: loss={selection_loss:.6f}')
                    if not math.isfinite(selection_loss):
                        raise FloatingPointError('Validation loss is non-finite')
                if selection_loss < best_loss:
                    best_loss = selection_loss
                    self._save_lora(output_dir, 'skelora')
                    print(f'[保存] 最佳 LoRA ({history['selection_metric']}) → {output_dir}')
            history['epochs'].append(record)
            history['best_selection_loss'] = best_loss
            history['best_fixed_probe_loss'] = min([initial_probe] + [item['fixed_probe_loss'] for item in history['epochs'] if 'fixed_probe_loss' in item])
            (Path(output_dir) / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        self._save_lora(os.path.join(output_dir, 'final'), 'skelora')
        print(f'[训练] 完成! 最佳 Loss={best_loss:.4f}')
        print(f'[训练] 推理配置 models.sketch_lora 应指向 {output_dir}, load_lora=true')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--kind', choices=('masked_source',), default='masked_source')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--split-manifest', type=Path, default=ROOT / 'train/split_manifest.json')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gate', type=float, default=0.1)
    parser.add_argument('--validation-samples', type=int, default=32)
    parser.add_argument('--initial-lora')
    parser.add_argument('--perceptual-weight', type=float, default=0.5)
    parser.add_argument('--perceptual-every', type=int, default=4)
    parser.add_argument('--perceptual-grid', type=int, choices=[4], default=4)
    parser.add_argument('--text-lora', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.perceptual_weight < 0 or args.perceptual_every < 1:
        raise ValueError('Invalid perceptual loss configuration')
    if not args.name or any((x in args.name for x in ('/', '\\', '..', ':'))):
        raise ValueError('Unsafe output name')
    output = Path(local_config.models.sketch_lora) / args.name
    if output.exists():
        raise ValueError('Existing output; refuse overwrite')
    cfg = {'data': {'train_dir': local_config.data_dir, 'split_manifest': str((ROOT / args.split_manifest).resolve()), 'image_size': 512, 'mask_edit_color': 'black', 'default_prompt': '', 'condition_image_kind': args.kind}, 'model': {'flux_base': local_config.models.flux_base, 'lora_output_dir': str(output), 'allow_base_relocation': True}, 'training': {'seed': 42, 'num_epochs': args.epochs, 'batch_size': 1, 'gradient_accumulation_steps': args.accumulation, 'learning_rate': args.lr, 'guidance': 3.5, 'weight_decay': 0.01, 'max_grad_norm': 1.0, 'save_every_epochs': args.epochs, 'mask_loss_weight': 1.0, 'condition_attention_scale': args.gate, 'image_lora': True, 'use_redux': False, 'background_lock': True, 'probe_every_epochs': 1 if args.epochs <= 3 else 5, 'validation_samples': args.validation_samples, 'text_dropout': 0.1, 'background_loss_weight': 0.05, 'reconstruction_weight': 0.05}, 'lora': {'rank': args.rank, 'alpha': args.rank, 'dropout': 0.0, 'target_modules': ['x_embedder', 'to_q', 'to_k', 'to_v', 'to_out.0', 'ff.net.2', 'proj_mlp', 'proj_out', 'norm1.linear', 'norm.linear']}}
    if args.initial_lora:
        cfg['model']['initial_lora'] = args.initial_lora
    if args.text_lora:
        cfg['lora']['target_modules'].extend(['context_embedder', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out', 'norm1_context.linear', 'ff_context.net.2'])
        cfg['model']['allow_text_lora_expansion'] = True
    cfg['training']['text_lora'] = args.text_lora
    cfg['training'].update(perceptual_weight=args.perceptual_weight, perceptual_every=args.perceptual_every, perceptual_grid=args.perceptual_grid)
    config_dir = ROOT / 'train/configs'
    config_dir.mkdir(exist_ok=True)
    path = config_dir / f'{args.name}.yaml'
    if path.exists():
        raise ValueError('Existing config; refuse overwrite')
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    trainer = VisualTrainer(load_config(str(path)), args.samples)
    trainer.train()
    if torch.cuda.is_available():
        print(f'[memory] peak allocated={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB', flush=True)
if __name__ == '__main__':
    main()
