"""Shared attention/LoRA routing, latent masks, source protection and CFG."""
from __future__ import annotations
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, Transformer2DModelOutput
from peft.tuners.tuners_utils import BaseTunerLayer
from accelerate.utils import is_torch_version
logger = logging.getLogger(__name__)
ATTENTION_PROTOCOL = 'condition_kv_probability_gate_v1'

def condition_attention_mask(query, key, base_key_count, scale, mask=None):
    scale = float(scale)
    if not math.isfinite(scale) or scale < 0 or scale > 1:
        raise ValueError('condition_attention_scale must be finite and between 0 and 1')
    if base_key_count < 1 or base_key_count > key.shape[-2]:
        raise ValueError('Invalid base key count')
    if scale == 1 or base_key_count == key.shape[-2]:
        return mask
    bias = torch.zeros((1, 1, 1, key.shape[-2]), device=query.device, dtype=query.dtype)
    bias[..., base_key_count:] = math.log(scale) if scale else float('-inf')
    if mask is not None:
        if mask.dtype == torch.bool:
            bias = bias.masked_fill(~mask, float('-inf'))
        else:
            bias = bias + mask
    return bias

def packed_hole_mask(mask, height, width, channels, pack_latents):
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError('Hole mask must have shape [B,1,H,W]')
    field = F.interpolate(mask.float(), size=(height, width), mode='area').clamp(0, 1)
    field = field.repeat(1, channels, 1, 1)
    return pack_latents(field, field.shape[0], channels, height, width)

def lock_source_trajectory(latents, source, noise, mask, sigma):
    sigma = torch.as_tensor(sigma, device=latents.device, dtype=latents.dtype)
    source_path = (1 - sigma) * source + sigma * noise
    return mask * latents + (1 - mask) * source_path

def mix_true_cfg(positive, negative, scale):
    if positive.shape != negative.shape or not math.isfinite(scale) or scale < 1:
        raise ValueError('Invalid true CFG inputs')
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise FloatingPointError('Non-finite CFG velocity')
    return (negative.float() + scale * (positive.float() - negative.float())).to(positive.dtype)

class enable_lora:
    def __init__(self, lora_modules: List[BaseTunerLayer], activated: bool) -> None:
        self.activated: bool = activated
        if activated:
            return
        self.lora_modules: List[BaseTunerLayer] = [each for each in lora_modules if isinstance(each, BaseTunerLayer)]
        self.scales = [{active_adapter: lora_module.scaling[active_adapter] for active_adapter in lora_module.active_adapters if active_adapter in lora_module.scaling} for lora_module in self.lora_modules]

    def __enter__(self) -> None:
        if self.activated:
            return
        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            lora_module.scale_layer(0)

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        if self.activated:
            return
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                if active_adapter not in lora_module.scaling:
                    continue
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]

class specify_lora:
    
    def __init__(self, lora_modules: List[BaseTunerLayer], specified_lora: str) -> None:
        self.specified_lora: str = specified_lora
        self.lora_modules: List[BaseTunerLayer] = [each for each in lora_modules if isinstance(each, BaseTunerLayer)]
        self.scales = [{active_adapter: lora_module.scaling[active_adapter] for active_adapter in lora_module.active_adapters if active_adapter in lora_module.scaling} for lora_module in self.lora_modules]

    def __enter__(self) -> None:
        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                if active_adapter not in lora_module.scaling:
                    continue
                lora_module.scaling[active_adapter] = 1 if active_adapter == self.specified_lora else 0

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                if active_adapter not in lora_module.scaling:
                    continue
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]

def attn_forward(attn: Attention, hidden_states: torch.FloatTensor, condition_names: List[str], encoder_hidden_states: torch.FloatTensor=None, condition_latents: torch.FloatTensor=None, attention_mask: Optional[torch.FloatTensor]=None, image_rotary_emb: Optional[torch.Tensor]=None, cond_rotary_emb: Optional[torch.Tensor]=None, model_config: Optional[Dict[str, Any]]=None, use_cache: bool=False) -> torch.FloatTensor:
    """
    文本-图像-条件 三流联合注意力

    计算流程:
    1. 图像流: hidden_states → Q_img, K_img, V_img (可选LoRA)
    2. 文本流: encoder_hidden_states → Q_txt, K_txt, V_txt
    3. 条件流: condition_latents → Q_cond, K_cond, V_cond (specify_lora)
    4. Concat(Q, K, V) → Joint Attention
    5. Split → 独立投影 → 返回各流更新值

    Args:
        attn:                 注意力模块
        hidden_states:        图像潜变量 [B, N_img, C]
        condition_names:      条件名称列表
        encoder_hidden_states: 文本潜变量 [B, N_txt, C]
        condition_latents:    条件潜变量列表 [[B, N_cond, C], ...]
        attention_mask:       注意力掩码
        image_rotary_emb:     图像旋转位置嵌入
        cond_rotary_emb:      条件旋转位置嵌入列表
        model_config:         模型配置 dict
        use_cache:            是否使用条件缓存

    Returns:
        更新后的 (hidden_states, encoder_hidden_states, condition_latents)
    """
    if model_config is None:
        model_config = {}
    hh_l = hidden_states.shape[1]
    batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    with enable_lora((attn.to_q, attn.to_k, attn.to_v), model_config.get('latent_lora', False)):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    if encoder_hidden_states is not None:
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)
    if not isinstance(attn.to_q, BaseTunerLayer):
        condition_latents = None
    if condition_latents is not None:
        cond_query, cond_key, cond_value = ([], [], [])
        for i, adapter_name in enumerate(condition_names):
            with specify_lora((attn.to_q, attn.to_k, attn.to_v), adapter_name):
                cond_query.append(attn.to_q(condition_latents[i]))
                cond_key.append(attn.to_k(condition_latents[i]))
                cond_value.append(attn.to_v(condition_latents[i]))
        cond_query = [each.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) for each in cond_query]
        cond_key = [each.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) for each in cond_key]
        cond_value = [each.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) for each in cond_value]
        if attn.norm_q is not None:
            cond_query = [attn.norm_q(each) for each in cond_query]
        if attn.norm_k is not None:
            cond_key = [attn.norm_k(each) for each in cond_key]
    if condition_latents is not None and cond_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb
        cond_query = [apply_rotary_emb(query, rotary_emb) for query, rotary_emb in zip(cond_query, cond_rotary_emb)]
        cond_key = [apply_rotary_emb(key, rotary_emb) for key, rotary_emb in zip(cond_key, cond_rotary_emb)]
        if use_cache:
            attn.cached = {'k': cond_key, 'v': cond_value}
    if condition_latents is not None:
        q_h = torch.cat([query], dim=2)
        k_h = torch.cat([key, *cond_key], dim=2)
        v_h = torch.cat([value, *cond_value], dim=2)
    else:
        q_h, k_h, v_h = (query, key, value)
    if condition_latents is None and hasattr(attn, 'cached') and use_cache:
        k_h = torch.cat([key, *attn.cached['k']], dim=2)
        v_h = torch.cat([value, *attn.cached['v']], dim=2)
    main_attention_mask = condition_attention_mask(q_h, k_h, key.shape[2], model_config.get('condition_attention_scale', 0.01), attention_mask)
    hidden_states = F.scaled_dot_product_attention(q_h, k_h, v_h, dropout_p=0.0, is_causal=False, attn_mask=main_attention_mask)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)
    t = []
    for i, _ in enumerate(condition_latents) if condition_latents is not None else []:
        independent = model_config.get('independent_condition', False)
        q_t = cond_query[i]
        k_t = torch.cat([key, cond_key[i]], dim=2) if not independent else cond_key[i]
        v_t = torch.cat([value, cond_value[i]], dim=2) if not independent else cond_value[i]
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, dropout_p=0.0, is_causal=False, attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = out.to(query.dtype)
        t.append(out)
    condition_latents = t or None
    if encoder_hidden_states is not None:
        ee = hidden_states[:, :encoder_hidden_states.shape[1]]
        hh = hidden_states[:, encoder_hidden_states.shape[1]:encoder_hidden_states.shape[1] + hh_l]
        encoder_hidden_states = ee
        hidden_states = hh
        with enable_lora((attn.to_out[0],), model_config.get('latent_lora', False)):
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        if condition_latents is not None:
            final_c_outs = []
            for i, (c, adapter_name) in enumerate(zip(condition_latents, condition_names)):
                with specify_lora((attn.to_out[0],), adapter_name):
                    c = attn.to_out[0](c)
                    c = attn.to_out[1](c)
                final_c_outs.append(c)
            condition_latents = final_c_outs
        return (hidden_states, encoder_hidden_states, condition_latents) if condition_latents is not None else (hidden_states, encoder_hidden_states)
    elif condition_latents is not None:
        hh = hidden_states[:, :hh_l]
        hidden_states = hh
        return (hidden_states, condition_latents)
    else:
        return hidden_states

def block_forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, condition_latents: torch.FloatTensor, temb: torch.FloatTensor, cond_temb: torch.FloatTensor, condition_names: List[str]=None, cond_rotary_emb=None, image_rotary_emb=None, model_config: Optional[Dict[str, Any]]=None, use_cache: bool=False):
    """
    双流 Transformer 块: 文本流和图像流分离处理，条件流并行交互

    结构:
        ┌─ Norm1 + 门控 ─→ Attention ─→ +残差 ─┐
        │  (文本/图像/条件 各自归一化)           │
        └─ Norm2 + MLP ──→ +残差 ─────────────┘
    """
    if model_config is None:
        model_config = {}
    if not isinstance(self.attn.to_q, BaseTunerLayer):
        condition_latents = None
    use_cond = condition_latents is not None
    with enable_lora((self.norm1.linear,), model_config.get('latent_lora', False)):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)
    if use_cond:
        norm_condition_latents, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = ([] for _ in range(5))
        for i, (each, adapter_name) in enumerate(zip(condition_latents, condition_names)):
            with specify_lora((self.norm1.linear,), adapter_name):
                res = self.norm1(each, emb=cond_temb)
            norm_condition_latents.append(res[0])
            cond_gate_msa.append(res[1])
            cond_shift_mlp.append(res[2])
            cond_scale_mlp.append(res[3])
            cond_gate_mlp.append(res[4])
    result = attn_forward(self.attn, model_config=model_config, hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states, condition_latents=norm_condition_latents if use_cond else None, image_rotary_emb=image_rotary_emb, cond_rotary_emb=cond_rotary_emb if use_cond else None, condition_names=condition_names, use_cache=use_cache)
    attn_output, context_attn_output = result[:2]
    cond_attn_output = result[2] if use_cond else None
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output
    if use_cond:
        cond_attn_output = [cgm.unsqueeze(1) * cao for cgm, cao in zip(cond_gate_msa, cond_attn_output)]
        condition_latents = [cl + cao for cl, cao in zip(condition_latents, cond_attn_output)]
        if model_config.get('add_cond_attn', False):
            hidden_states += cond_attn_output
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    if use_cond:
        norm_condition_latents = [self.norm2(each) for each in condition_latents]
        norm_condition_latents = [each * (1 + csm1[:, None]) + csm[:, None] for each, csm1, csm in zip(norm_condition_latents, cond_scale_mlp, cond_shift_mlp)]
    with enable_lora((self.ff.net[2],), model_config.get('latent_lora', False)):
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output
    if use_cond:
        cond_ff_output = []
        for i, adapter_name in enumerate(condition_names):
            with specify_lora((self.ff.net[2],), adapter_name):
                res = self.ff(norm_condition_latents[i])
                res = cond_gate_mlp[i].unsqueeze(1) * res
            cond_ff_output.append(res)
    hidden_states = hidden_states + ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output
    if use_cond:
        condition_latents = [cl + cfo for cl, cfo in zip(condition_latents, cond_ff_output)]
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    return (encoder_hidden_states, hidden_states, condition_latents if use_cond else None)

def single_block_forward(self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor, condition_names: List[str]=None, image_rotary_emb=None, condition_latents: torch.FloatTensor=None, cond_temb: torch.FloatTensor=None, cond_rotary_emb=None, model_config: Optional[Dict[str, Any]]=None, use_cache: bool=False):
    """
    单流 Transformer 块: 文本+图像合并为一个统一流，条件流并行交互

    与 dual block 的区别:
    - 文本和图像不分开，直接 concat 后一起处理
    - 条件仍然独立编码和更新
    """
    if model_config is None:
        model_config = {}
    if not isinstance(self.attn.to_q, BaseTunerLayer):
        condition_latents = None
    using_cond = condition_latents is not None
    residual = hidden_states
    with enable_lora((self.norm.linear, self.proj_mlp), model_config.get('latent_lora', False)):
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    if using_cond:
        residual_cond = condition_latents
        norm_condition_latents, cond_gate = ([], [])
        for each, adapter_name in zip(condition_latents, condition_names):
            with specify_lora((self.norm.linear,), adapter_name):
                res = self.norm(each, emb=cond_temb)
            norm_condition_latents.append(res[0])
            cond_gate.append(res[1])
        mlp_cond_hidden_states = []
        for i, adapter_name in enumerate(condition_names):
            with specify_lora((self.proj_mlp,), adapter_name):
                res = self.act_mlp(self.proj_mlp(norm_condition_latents[i]))
            mlp_cond_hidden_states.append(res)
    attn_output = attn_forward(self.attn, model_config=model_config, hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb, condition_names=condition_names, use_cache=use_cache, **{'condition_latents': norm_condition_latents, 'cond_rotary_emb': cond_rotary_emb if using_cond else None} if using_cond else {})
    if using_cond:
        attn_output, cond_attn_output = attn_output
    with enable_lora((self.proj_out,), model_config.get('latent_lora', False)):
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
    if using_cond:
        condition_latents = [torch.cat([cao, mlp], dim=2) for cao, mlp in zip(cond_attn_output, mlp_cond_hidden_states)]
        cond_gate = [each.unsqueeze(1) for each in cond_gate]
        t = []
        for i, adapter_name in enumerate(condition_names):
            with specify_lora((self.proj_out,), adapter_name):
                res = cond_gate[i] * self.proj_out(condition_latents[i])
                t.append(res)
        condition_latents = t
        condition_latents = [rc + cl for rc, cl in zip(residual_cond, condition_latents)]
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return (hidden_states, condition_latents) if using_cond else (hidden_states, None)

def prepare_params(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor=None, pooled_projections: torch.Tensor=None, timestep: torch.LongTensor=None, img_ids: torch.Tensor=None, txt_ids: torch.Tensor=None, guidance: torch.Tensor=None, joint_attention_kwargs: Optional[Dict[str, Any]]=None, return_dict: bool=True, **kwargs):
    """将 transformer_forward 的 **params 解包为命名变量"""
    return (hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance, joint_attention_kwargs, return_dict)

def transformer_forward(transformer: FluxTransformer2DModel, condition_latents: torch.Tensor, condition_ids: torch.Tensor, condition_type_ids: torch.Tensor, condition_names: List[str], model_config: Optional[Dict[str, Any]]=None, c_t: float=0, use_cache: bool=False, **params):
    """
    多条件 Transformer 前向传播

    在原生 FLUX Transformer 的基础上:
    1. 为条件 Token 添加独立的 x_embedder 编码 (使用 specify_lora)
    2. 为条件 Token 生成独立的位置嵌入
    3. 在每个双流/单流块中注入条件，实现联合推理

    Args:
        transformer:         FLUX Transformer 模型
        condition_latents:   条件 Token 列表 [[B, N, C], ...]
        condition_ids:       条件位置 ID 列表
        condition_type_ids:  条件类型 ID 列表
        condition_names:     条件名称列表 (用于 LoRA 路由)
        model_config:        模型配置
        c_t:                 条件时间步系数 (默认 0)
        use_cache:           是否使用条件缓存
        **params:            原生 transformer 参数

    Returns:
        Transformer2DModelOutput 或 (output,) tuple
    """
    if model_config is None:
        model_config = {}
    self = transformer
    use_condition = condition_latents is not None
    hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance, joint_attention_kwargs, return_dict = prepare_params(**params)
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop('scale', 1.0)
        for k, v in joint_attention_kwargs.items():
            model_config.setdefault(k, v)
    else:
        lora_scale = 1.0
    with enable_lora((self.x_embedder,), model_config.get('latent_lora', False)):
        hidden_states = self.x_embedder(hidden_states)
    if use_condition:
        t = []
        for condition_latent, adapter_name in zip(condition_latents, condition_names):
            with specify_lora((self.x_embedder,), adapter_name):
                t.append(self.x_embedder(condition_latent))
        condition_latents = t
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None
    temb = self.time_text_embed(timestep, pooled_projections) if guidance is None else self.time_text_embed(timestep, guidance, pooled_projections)
    cond_temb = self.time_text_embed(torch.ones_like(timestep) * c_t * 1000, pooled_projections) if guidance is None else self.time_text_embed(torch.ones_like(timestep) * c_t * 1000, guidance, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    if txt_ids.ndim == 3:
        logger.warning('Passing `txt_ids` 3d torch.Tensor is deprecated.Please remove the batch dimension and pass it as a 2d torch Tensor')
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning('Passing `img_ids` 3d torch.Tensor is deprecated.Please remove the batch dimension and pass it as a 2d torch Tensor')
        img_ids = img_ids[0]
    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)
    if use_condition:
        cond_rotary_emb = [self.pos_embed(condition_id) for condition_id in condition_ids]
    for index_block, block in enumerate(self.transformer_blocks):
        if self.training and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = {'use_reentrant': False} if is_torch_version('>=', '1.11.0') else {}
            encoder_hidden_states, hidden_states, condition_latents = torch.utils.checkpoint.checkpoint(block_forward, self=block, model_config=model_config, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, condition_latents=condition_latents if use_condition else None, temb=temb, cond_temb=cond_temb if use_condition else None, cond_rotary_emb=cond_rotary_emb if use_condition else None, condition_names=condition_names if use_condition else None, image_rotary_emb=image_rotary_emb, use_cache=use_cache, **ckpt_kwargs)
        else:
            encoder_hidden_states, hidden_states, condition_latents = block_forward(block, model_config=model_config, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, condition_latents=condition_latents if use_condition else None, temb=temb, cond_temb=cond_temb if use_condition else None, cond_rotary_emb=cond_rotary_emb if use_condition else None, condition_names=condition_names if use_condition else None, image_rotary_emb=image_rotary_emb, use_cache=use_cache)
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    for index_block, block in enumerate(self.single_transformer_blocks):
        if self.training and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = {'use_reentrant': False} if is_torch_version('>=', '1.11.0') else {}
            result = torch.utils.checkpoint.checkpoint(single_block_forward, self=block, model_config=model_config, hidden_states=hidden_states, temb=temb, image_rotary_emb=image_rotary_emb, use_cache=use_cache, **{'condition_latents': condition_latents, 'cond_temb': cond_temb, 'cond_rotary_emb': cond_rotary_emb, 'condition_names': condition_names} if use_condition else {}, **ckpt_kwargs)
        else:
            result = single_block_forward(block, model_config=model_config, hidden_states=hidden_states, temb=temb, image_rotary_emb=image_rotary_emb, use_cache=use_cache, **{'condition_latents': condition_latents, 'cond_temb': cond_temb, 'cond_rotary_emb': cond_rotary_emb, 'condition_names': condition_names} if use_condition else {})
        if use_condition:
            hidden_states, condition_latents = result
        else:
            hidden_states = result[0]
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
