import math
import secrets
import numpy as np
import torch
from diffusers.pipelines import FluxPipeline
from .before import Condition, config, prepare_images, visual_condition
from .processing import transformer_forward, packed_hole_mask, lock_source_trajectory, mix_true_cfg


def run_edit(source, mask, prompt, model_provider, steps=28, cfg=2., seed=42, random_seed=False):
    source, condition, field, preview, prompt = prepare_images(source, mask, prompt)
    if int(steps) != steps or not 10 <= steps <= 40:
        raise ValueError("Steps must be an integer between 10 and 40")
    if not math.isfinite(float(cfg)) or not 1 <= cfg <= 4:
        raise ValueError("Text guidance must be between 1 and 4")
    seed = secrets.randbelow(2147483647) if random_seed else int(seed)
    if not 0 <= seed <= 2147483647:
        raise ValueError("Seed is outside the supported range")
    pipe = model_provider()
    carrier = visual_condition(source, condition, field)
    with torch.inference_mode():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        result = pipe(prompt=prompt, base_image=source, sketch_image=carrier,
                      coefficient_map=field, num_inference_steps=int(steps),
                      condition_attention_scale=pipe.condition_attention_scale,
                      true_cfg_scale=float(cfg),
                      generator=torch.Generator(device="cuda").manual_seed(seed))
    record = {"prompt": prompt, "seed": seed, "steps": int(steps), "cfg": float(cfg),
              "weights": config.models.sketch_lora, "condition_kind": "masked_source",
              "edit_area_fraction": float(field.mean()), "stitch": False, "fit_scale": False}
    return result, source, preview, prompt, record


class OminiSketchPipeline(FluxPipeline):

    def __init__(self, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, transformer, scheduler):
        super().__init__(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                         text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
                         transformer=transformer, scheduler=scheduler)
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.default_sample_size = 64

    @torch.no_grad()
    def __call__(self, prompt, base_image, sketch_image, coefficient_map,
                 height=512, width=512, num_inference_steps=28, guidance_scale=3.5,
                 condition_attention_scale=.1, true_cfg_scale=2., generator=None):
        device = self._execution_device

        # Text sequence/global features; the second pass holds image conditions fixed.
        prompt_embeds, pooled, text_ids = self.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=device, num_images_per_prompt=1)
        negative_embeds = negative_pooled = negative_ids = None
        if true_cfg_scale > 1:
            negative_embeds, negative_pooled, negative_ids = self.encode_prompt(
                prompt="", prompt_2="", device=device, num_images_per_prompt=1)
        prompt_embeds = prompt_embeds * .5 * 2.0
        if negative_embeds is not None:
            negative_embeds = negative_embeds * .5 * 2.0

        # Preserve diffusers 0.31 size conventions and the trained 2x2 token packing.
        channels = self.transformer.config.in_channels // 4
        latents, image_ids = self.prepare_latents(
            1, channels, height // 2, width // 2, prompt_embeds.dtype,
            device, generator, None)
        tokens, ids, type_ids = Condition("sketch", sketch_image.resize((width, height))).encode(self)
        tokens = tokens * 1.0
        condition_latents, condition_ids = [tokens], [ids]
        condition_type_ids, condition_names = [type_ids], ["skelora"]
        scheduler_kwargs = {}
        if getattr(self.scheduler.config, "use_dynamic_shifting", False):
            scheduler_kwargs["mu"] = 0.0
        self.scheduler.set_timesteps(num_inference_steps, device=device, **scheduler_kwargs)

        # Source trajectory and edit mask use the same latent layout/noise as generation.
        noise = latents.clone()
        side = int(round(latents.shape[1] ** .5)) * 2
        source = Condition("subject", base_image.resize((side*self.vae_scale_factor,) * 2),
                           position_delta=[0, 0]).encode(self)[0].to(latents.dtype)
        mask_tensor = torch.as_tensor(np.clip(coefficient_map, 0, 1), device=device,
                                      dtype=torch.float32)[None, None]
        edit_mask = packed_hole_mask(mask_tensor, side, side, channels, self._pack_latents).to(latents.dtype)
        # Keep the existing arithmetic, including fractional mask-edge rounding.
        latents = (1-edit_mask)*((1-1.0)*source+1.0*noise)+edit_mask*noise
        self.vae.to("cpu")
        self.text_encoder.to("cpu")
        self.text_encoder_2.to("cpu")
        self.transformer.to(self._execution_device)

        # Predict velocity, apply true CFG, step the scheduler, then protect the background.
        attention = {"condition_attention_scale": condition_attention_scale, "latent_lora": True}
        for index, timestep_value in enumerate(self.scheduler.timesteps):
            timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
            guidance = None
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device, dtype=latents.dtype).expand(latents.shape[0])
            shared = dict(model_config=attention, condition_latents=condition_latents,
                          condition_ids=condition_ids, condition_type_ids=condition_type_ids,
                          condition_names=condition_names, hidden_states=latents,
                          timestep=timestep/1000, guidance=guidance, img_ids=image_ids,
                          return_dict=False, use_cache=False)
            velocity = transformer_forward(self.transformer, pooled_projections=pooled,
                                           encoder_hidden_states=prompt_embeds, txt_ids=text_ids, **shared)[0]
            if negative_embeds is not None:
                negative = transformer_forward(self.transformer, pooled_projections=negative_pooled,
                                               encoder_hidden_states=negative_embeds, txt_ids=negative_ids, **shared)[0]
                velocity = mix_true_cfg(velocity, negative, true_cfg_scale)
            latents = self.scheduler.step(velocity, timestep_value, latents, return_dict=False)[0]
            latents = lock_source_trajectory(latents, source, noise, edit_mask, self.scheduler.sigmas[index+1])

        # Direct VAE decoding: no RGB stitching, fitting into a hole or contour resizing.
        latents = self._unpack_latents(latents, height//2, width//2, self.vae_scale_factor)
        latents = latents/self.vae.config.scaling_factor+self.vae.config.shift_factor
        decoded = self.vae.decode(latents, return_dict=False)[0]
        return self.image_processor.postprocess(decoded, output_type="pil")[0]
