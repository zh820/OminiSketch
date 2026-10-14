"""Training accessories: configuration, dataset preparation, preencoding, losses and splits."""
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from core.before import visual_condition
LATENT_SIZE = 64

# Convert dataset masks without guessing their polarity.
def binary_hole_condition(mask, target_size=(512, 512), dilate_px=0):
    from scipy.ndimage import binary_dilation
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Hole mask must be two-dimensional')
    if dilate_px < 0:
        raise ValueError('dilate_px must be non-negative')
    if dilate_px:
        mask = binary_dilation(mask, iterations=dilate_px)
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask = np.asarray(img.resize(target_size, Image.Resampling.BILINEAR)) > 127
    field = mask.astype(np.float32)
    return (Image.fromarray(mask.astype(np.uint8) * 255).convert('RGB'), field)

def dataset_hole_condition(mask_image, target_size=(512, 512), edit_color='black'):
    if edit_color not in ('black', 'white'):
        raise ValueError('mask_edit_color must be black or white')
    values = np.asarray(mask_image.convert('L'))
    holes = values < 128 if edit_color == 'black' else values >= 128
    return binary_hole_condition(holes, target_size, dilate_px=0)

# Partition source groups and validate manifest coverage before training.
def grouped_partition(groups, seed=42, ratios=(0.7, 0.2, 0.1)):
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-08 or min(ratios) <= 0:
        raise ValueError('Expected three positive ratios summing to one')
    items = list(groups.values())
    random.Random(seed).shuffle(items)
    items.sort(key=len, reverse=True)
    total = sum(map(len, items))
    partitions = {'train': [], 'validation': [], 'test': []}
    targets = dict(zip(partitions, [total * ratio for ratio in ratios]))
    for members in items:
        name = max(partitions, key=lambda key: targets[key] - len(partitions[key]))
        partitions[name].extend(members)
    return {name: sorted(members) for name, members in partitions.items()}

def validate_manifest(manifest, discovered, data_root):
    root = Path(data_root).resolve()
    seen = set()
    partitions = {}
    for name in ('train', 'validation', 'test'):
        members = manifest['splits'][name]
        if not members:
            raise ValueError(f'Empty {name} partition')
        paths = []
        for member in members:
            path = (root / member).resolve()
            if not path.is_relative_to(root) or path == root:
                raise ValueError('Split path outside dataset')
            if path in seen:
                raise ValueError('Duplicate sample or cross-partition overlap')
            seen.add(path)
            paths.append(path)
        partitions[name] = paths
    if seen != {Path(path).resolve() for path in discovered}:
        raise ValueError('Manifest must cover exactly the discovered dataset')
    for group in manifest.get('source_groups', {}).values():
        owners = {name for name, members in manifest['splits'].items() if any((member in members for member in group))}
        if len(owners) != 1:
            raise ValueError('Source group leaks across partitions')
    return partitions

# Loss helpers preserve spatial layout and normalize by edit-region area.
def spatial_teacher_features(tokens, grid=4):
    if tokens.ndim != 3 or grid < 1:
        raise ValueError('Teacher tokens must have batch, spatial and feature axes')
    batch, count, channels = tokens.shape
    side = math.isqrt(count)
    if side * side != count or grid > side:
        raise ValueError('Teacher requires a square patch grid without a class token')
    features = tokens.float().transpose(1, 2).reshape(batch, channels, side, side)
    features = torch.nn.functional.adaptive_avg_pool2d(features, (grid, grid))
    features = features.flatten(2).transpose(1, 2)
    return torch.nn.functional.normalize(features, dim=-1)

def unpack_visual_tokens(tokens, height=64, width=64):
    if tokens.ndim != 3 or height % 2 or width % 2 or tokens.shape[-1] % 4:
        raise ValueError('Invalid packed visual token layout')
    batch, count, features = tokens.shape
    if count != height // 2 * (width // 2):
        raise ValueError('Packed token count does not match latent dimensions')
    channels = features // 4
    return tokens.reshape(batch, height // 2, width // 2, channels, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)

def region_bbox(field, padding=16, min_side=64):
    mask = np.asarray(field) > 0.5
    if mask.ndim != 2 or not mask.any():
        raise ValueError('Region bbox requires a nonempty 2D mask')
    ys, xs = np.where(mask)
    height, width = mask.shape
    w = min(width, max(min_side, int(xs.max() - xs.min() + 1) + 2 * padding))
    h = min(height, max(min_side, int(ys.max() - ys.min() + 1) + 2 * padding))
    left = max(0, min(width - w, int(round((xs.min() + xs.max() + 1 - w) / 2))))
    top = max(0, min(height - h, int(round((ys.min() + ys.max() + 1 - h) / 2))))
    return (left, top, left + w, top + h)

def area_normalized_loss(prediction, target, mask, background_weight=0.05):
    if prediction.shape != target.shape or mask.shape != prediction.shape:
        raise ValueError('Prediction, target and packed mask must have identical shapes')
    if not 0 <= background_weight <= 1:
        raise ValueError('Invalid background loss weight')
    error = (prediction.float() - target.float()).square()
    mask = mask.float()
    if not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError('Invalid mask values')
    edit = (error * mask).sum() / mask.sum().clamp_min(1)
    keep = 1 - mask
    background = (error * keep).sum() / keep.sum().clamp_min(1)
    return edit + background_weight * background

class DotDict(dict):
    """普通 dict 的点号访问包装: cfg.section.key"""

    def __getattr__(self, key):
        value = self.get(key)
        if value is None:
            raise AttributeError(f"配置中缺少 '{key}' 项")
        return DotDict(value) if isinstance(value, dict) else value

def load_config(config_path: str) -> DotDict:
    """加载训练配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'训练配置不存在: {config_path}\n请先编辑 train/train_config.yaml (所有训练参数都在其中)')
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    project_root = Path(__file__).resolve().parents[1]
    for section, key in (('data', 'train_dir'), ('model', 'flux_base'), ('model', 'lora_output_dir'), ('model', 'redux_image_encoder'), ('model', 'redux_image_embedder'), ('model', 'initial_lora'), ('data', 'split_manifest')):
        if key not in data[section]:
            continue
        value = Path(data[section][key]).expanduser()
        if not value.is_absolute():
            value = project_root / value
        data[section][key] = str(value.resolve())
    return DotDict(data)

def make_image_ids(n_tokens: int, device, dtype) -> torch.Tensor:
    """生成打包 latent 的位置 ID [n_tokens, 3]"""
    h = w = int(n_tokens ** 0.5)
    ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
    ids[..., 1] += torch.arange(h, device=device)[:, None]
    ids[..., 2] += torch.arange(w, device=device)[None, :]
    return ids.reshape(-1, 3)

def pil_to_tensor(image: Image.Image, size: int=512) -> torch.Tensor:
    """PIL → [1,3,H,W] ∈ [-1,1] (与 VAE 预处理后的数值范围一致)"""
    image = image.resize((size, size)).convert('RGB')
    arr = np.array(image).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
    return t.unsqueeze(0)

class SketchLoRADataset(Dataset):
    """
    掩码指令编辑数据集 (每样本一个目录)

    {data_dir}/
    └── <切片目录>/            # 名字任意, 自动递归扫描 (如 0-99/, 100-199/, ...)
        └── <样本目录>/
            ├── source.png   # 编辑前原图 (必需 → 复合输入公式的 x0_src)
            ├── mask.png     # 原始涂鸦区域 (默认黑色为编辑区, 按配置转换极性)
            ├── target.png   # 编辑后目标图 (训练目标)
            └── prompt.txt   # 编辑指令 (可选, 缺省 default_prompt)
    """

    def __init__(self, data_dir: str, image_size: int=512, default_prompt: str='', mask_edit_color: str='black'):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.default_prompt = default_prompt
        self.mask_edit_color = mask_edit_color
        if not self.data_dir.exists():
            raise FileNotFoundError(f'数据目录不存在: {self.data_dir}\n请按以下结构准备训练数据:\n  {self.data_dir}/<切片目录>/<样本目录>/\n      source.png  编辑前原图 (必需)\n      mask.png    编辑区域掩码\n      target.png  编辑后目标图\n      prompt.txt  编辑指令 (可选)')
        self.sample_dirs = []
        skipped = 0
        for mask_path in sorted(self.data_dir.rglob('mask.png')):
            sample_dir = mask_path.parent
            if not (sample_dir / 'target.png').exists():
                print(f'[数据集] 跳过缺少 target.png 的样本目录: {sample_dir}')
                skipped += 1
                continue
            if not (sample_dir / 'source.png').exists():
                print(f'[数据集] 跳过缺少 source.png 的样本目录: {sample_dir}')
                skipped += 1
                continue
            self.sample_dirs.append(sample_dir)
        if not self.sample_dirs:
            raise FileNotFoundError(f'{self.data_dir} 下没有找到 source.png + mask.png + target.png 配对的样本目录')
        n_prompts = sum((1 for d in self.sample_dirs if (d / 'prompt.txt').exists()))
        print(f'[数据集] {len(self.sample_dirs)} 个训练样本 (跳过 {skipped} 个缺文件)')
        print(f'[数据集] {n_prompts} 个样本带编辑指令 (其余用 default_prompt)')

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        target = pil_to_tensor(Image.open(sample_dir / 'target.png'), self.image_size)
        with Image.open(sample_dir / 'source.png') as source_image:
            source_image = source_image.convert('RGB').resize((self.image_size, self.image_size))
        source = pil_to_tensor(source_image, self.image_size)
        with Image.open(sample_dir / 'mask.png') as mask_img:
            sketch_img, hole_map = dataset_hole_condition(mask_img, (self.image_size, self.image_size), self.mask_edit_color)
        sketch = pil_to_tensor(sketch_img, self.image_size)
        prompt = self.default_prompt
        prompt_path = sample_dir / 'prompt.txt'
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding='utf-8').strip()
        sample = {'sketch': sketch, 'hole_map': torch.from_numpy(hole_map)[None, None], 'target': target, 'source': source, 'source_image': source_image, 'prompt': prompt, 'filename': sample_dir.relative_to(self.data_dir).as_posix()}
        field = sample['hole_map'][0, 0].numpy()
        mask = Image.fromarray((field * 255).astype(np.uint8)).convert('RGB')
        condition = visual_condition(sample['source_image'], mask, field, 'masked_source')
        sample['sketch'] = pil_to_tensor(condition, self.image_size)
        return sample

class TrainingPreprocessor:
    """Use the trainer's loaded models to build image, text and teacher caches."""

    @torch.no_grad()
    def _encode_latent(self, image: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W]∈[-1,1] → VAE latent [B,16,64,64] (含 shift/scale 归一)"""
        image = image.to(self.device).to(self.dtype)
        latent = self.pipe.vae.encode(image).latent_dist.sample()
        latent = (latent - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        return latent

    def _to_tokens(self, latent: torch.Tensor) -> torch.Tensor:
        """[B,16,64,64] latent → 打包 token [B,1024,64]"""
        return self.pipe._pack_latents(latent, latent.shape[0], latent.shape[1], latent.shape[2], latent.shape[3])

    def _make_condition(self, tokens: torch.Tensor, name: str, type_id: int) -> tuple:
        """构造单个条件的三件套 (与 core/transformer.py 的输入约定一致)"""
        ids = make_image_ids(tokens.shape[1], self.device, self.dtype)
        type_ids = torch.ones(tokens.shape[1], 1, device=self.device, dtype=self.dtype) * type_id
        return (tokens, ids, type_ids, name)

    @torch.no_grad()
    def _preencode_base(self, sample: dict) -> dict:
        """单个样本 → 打包 token 缓存 (全部 .cpu())"""
        cached = {'filename': sample['filename'], 'x0_tokens': self._to_tokens(self._encode_latent(sample['target'])).cpu(), 'x0_src_tokens': self._to_tokens(self._encode_latent(sample['source'])).cpu(), 'sketch_tokens': self._to_tokens(self._encode_latent(sample['sketch'])).cpu()}
        mask_map = sample['hole_map'].float()
        mask_map = F.interpolate(mask_map, size=(LATENT_SIZE, LATENT_SIZE), mode='area')
        cached['mask_weight_map'] = mask_map.cpu()
        prompt_embeds, pooled, text_ids = self.pipe.encode_prompt(prompt=sample['prompt'], prompt_2=sample['prompt'], device=self.device, num_images_per_prompt=1)
        if text_ids.ndim == 3:
            text_ids = text_ids[0]
        cached['prompt_embeds'] = prompt_embeds.cpu()
        cached['pooled'] = pooled.cpu()
        cached['text_ids'] = text_ids.cpu()
        return cached

    def _visual_features(self, pixels, bbox):
        left, top, right, bottom = bbox
        crop = F.interpolate(pixels[:, :, top:bottom, left:right].float(), size=(384, 384), mode='bicubic', align_corners=False).clamp(-1, 1)
        result = self.vision_teacher(crop.to(self.device, self.dtype))
        grid = self.cfg.training.get('perceptual_grid', 0)
        if grid:
            return spatial_teacher_features(result.last_hidden_state, grid)
        features = result.pooler_output.float()
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def _preencode(self, sample):
        result = self._preencode_base(sample)
        if self.empty_text is None:
            embeds, pooled, ids = self.pipe.encode_prompt(prompt='', prompt_2='', device=self.device, num_images_per_prompt=1)
            self.empty_text = (embeds.cpu(), pooled.cpu(), ids.cpu())
        if self.vision_teacher is not None:
            bbox = region_bbox(sample['hole_map'][0, 0].numpy())
            result['region_bbox'] = bbox
            result['target_visual_feature'] = self._visual_features(sample['target'], bbox).cpu()
        return result
