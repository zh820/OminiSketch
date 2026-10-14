from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml
from PIL import Image
CONDITION_PROTOCOL = 'binary_hole_v1'

class Config:
    
    _instance: Optional['Config'] = None
    _data: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[str]=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: Optional[str]=None):
        """加载 YAML 配置文件"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config.yaml'
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f'配置文件不存在: {config_path}\n请确保在项目根目录下创建了 config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f)
        project_root = Path(__file__).resolve().parents[1]
        for key in ('flux_base', 'sketch_lora', 'redux_image_encoder'):
            value = self._data.get('models', {}).get(key)
            if value:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = project_root / path
                self._data['models'][key] = str(path.resolve())
        if self._data.get('data_dir'):
            path = Path(self._data['data_dir']).expanduser()
            if not path.is_absolute():
                path = project_root / path
            self._data['data_dir'] = str(path.resolve())

    def __getattr__(self, key: str) -> Any:
        """支持 config.section.key 的访问方式"""
        if key.startswith('_'):
            raise AttributeError(key)
        data = self._data.get(key)
        if data is None:
            raise AttributeError(f"配置中没有 '{key}' 节")
        if isinstance(data, dict):
            return _ConfigSection(data)
        return data

    def get(self, key: str, default: Any=None) -> Any:
        """安全获取配置项"""
        return self._data.get(key, default)

    def __repr__(self) -> str:
        return f'Config(sections={list(self._data.keys())})'

class _ConfigSection:
    """配置节的点号访问包装器"""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            raise AttributeError(key)
        if key not in self._data:
            raise AttributeError(f"配置节中没有 '{key}' 项")
        value = self._data[key]
        if isinstance(value, dict):
            return _ConfigSection(value)
        return value

    def get(self, key: str, default: Any=None) -> Any:
        return self._data.get(key, default)

    def __repr__(self) -> str:
        return f'ConfigSection({self._data})'

def load_config(config_path: Optional[str]=None) -> Config:
    """显式加载/重载配置"""
    Config._instance = None
    return Config(config_path)
config = Config()

def prepare_images(source, mask, prompt):
    """Normalize source, black-edit mask and text; return the five pipeline inputs."""
    if not isinstance(source, Image.Image):
        raise ValueError('A source image is required')
    
    if source.width * source.height > 16000000:
        raise ValueError('Source image exceeds 16 million pixels')
    
    prompt = (prompt or '').strip()
    
    if not prompt:
        raise ValueError('An edit instruction is required')
    if not isinstance(mask, Image.Image):
        raise ValueError('An edit mask is required')
    
    source = source.convert('RGB').resize((512, 512))
    
    # Uploaded/UI masks use black for editing; internal conditions use white.
    holes = np.asarray(mask.convert('L')) < 128
    hole_image = Image.fromarray(holes.astype(np.uint8) * 255)
    holes = np.asarray(hole_image.resize((512, 512), Image.Resampling.BILINEAR)) > 127
    field = holes.astype(np.float32)
    condition = Image.fromarray(holes.astype(np.uint8) * 255).convert('RGB')
    if not np.any(field > 0.5):
        raise ValueError('The edit region is empty')
    
    preview = Image.fromarray(np.where(field > 0.5, 0, 255).astype(np.uint8)).convert('RGB')
    
    return (source, condition, field, preview, prompt)

def prepare_editor(editor, mask_image, mode, prompt):

    if not isinstance(editor, dict) or not isinstance(editor.get('background'), Image.Image):
        raise ValueError('A source image is required')
    
    background = editor['background']
    
    if background.width * background.height > 16000000:
        raise ValueError('Source image exceeds 16 million pixels')
    
    if mode == 1:
        mask = mask_image
    elif mode == 0:
    
        painted = np.zeros((background.height, background.width), dtype=bool)
    
        for layer in editor.get('layers') or []:
            if not isinstance(layer, Image.Image) or layer.mode != 'RGBA':
                raise ValueError('Drawing layers must retain their alpha channel')
            
            layer = layer.resize(background.size, Image.Resampling.NEAREST)
            
            painted |= np.asarray(layer)[..., 3] > 0
        
        # Only painted pixels are editable; do not fill contours or expand strokes.
        mask = Image.fromarray(np.where(painted, 0, 255).astype(np.uint8)).convert('RGB')
    
    else:
        raise ValueError('Invalid region mode')
    
    return prepare_images(background, mask, prompt)

class Condition:
    """Encode the spatial background or source through the existing FLUX VAE."""

    def __init__(self, condition_type, raw_img, position_delta=None):
        self.condition = raw_img
        self.condition_type = condition_type
        self.position_delta = position_delta

    def encode(self, pipeline):
        import torch
        tokens, ids = self._encode_images(pipeline, self.condition)
        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]
        type_id = 12 if self.condition_type == 'sketch' else 4
        return (tokens, ids, torch.ones_like(ids[:, :1]) * type_id)

    @staticmethod
    def _encode_images(pipeline, images):
        """通过 VAE 编码图像为潜空间 Token"""
        import torch
        
        images = pipeline.image_processor.preprocess(images)
        exec_device = pipeline._execution_device
        images = images.to(exec_device).to(pipeline.vae.dtype)
        images = pipeline.vae.encode(images).latent_dist.sample()
        images = (images - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
        images_tokens = pipeline._pack_latents(images, *images.shape)
        n_tokens = images_tokens.shape[1]
        
        h = w = int(n_tokens ** 0.5)
        
        images_ids = torch.zeros(h, w, 3, device=pipeline.device, dtype=pipeline.dtype)
        images_ids[..., 1] = images_ids[..., 1] + torch.arange(h, device=pipeline.device)[:, None]
        images_ids[..., 2] = images_ids[..., 2] + torch.arange(w, device=pipeline.device)[None, :]
        images_ids = images_ids.reshape(h * w, 3)
        return (images_tokens, images_ids)

def visual_condition(source, condition, field, kind='masked_source'):
    """Retain background RGB and replace the edit region with neutral gray."""
    pixels = np.asarray(source.convert('RGB').resize(condition.size), dtype=np.float32)
    mask = np.asarray(field, dtype=np.float32)
    if mask.shape != pixels.shape[:2]:
        raise ValueError('Source and mask dimensions differ')
    pixels = pixels * (1 - mask[..., None]) + 127.5 * mask[..., None]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

def load_model():
    """Load configured local resources; leave file/schema checks to their loaders."""
    import torch
    from diffusers import AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    from peft import PeftModel
    from peft.tuners.tuners_utils import BaseTunerLayer
    from .pipeline import OminiSketchPipeline
    
    weights = Path(config.models.sketch_lora)
    
    if (weights / 'skelora').is_dir():
        weights = weights / 'skelora'
    
    metadata = json.loads((weights / 'training_protocol.json').read_text(encoding='utf-8'))
    
    flux = config.models.flux_base
    
    shared = dict(torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    
    pipe = OminiSketchPipeline(vae=AutoencoderKL.from_pretrained(flux, subfolder='vae', **shared), text_encoder=CLIPTextModel.from_pretrained(flux, subfolder='text_encoder', **shared), tokenizer=CLIPTokenizer.from_pretrained(flux, subfolder='tokenizer', local_files_only=True), text_encoder_2=T5EncoderModel.from_pretrained(flux, subfolder='text_encoder_2', **shared), tokenizer_2=T5TokenizerFast.from_pretrained(flux, subfolder='tokenizer_2', local_files_only=True), transformer=FluxTransformer2DModel.from_pretrained(flux, subfolder='transformer', **shared), scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(flux, subfolder='scheduler', local_files_only=True))
    
    pipe.enable_model_cpu_offload()
    
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(weights), adapter_name='skelora')
    
    for module in pipe.transformer.base_model.modules():
        if isinstance(module, BaseTunerLayer):
            module.set_adapter(['skelora'], inference_mode=True)
    
    pipe.transformer.to('cpu')
    torch.cuda.empty_cache()
    pipe.transformer.eval().requires_grad_(False)
    pipe.condition_attention_scale = metadata['condition_attention_scale']
    return pipe
