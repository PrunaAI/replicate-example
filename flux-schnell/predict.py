# Prediction interface for Cog ⚙️
# https://cog.run/python

import tempfile

import torch
from cog import BasePredictor, Input, Path
from diffusers import DPMSolverMultistepScheduler, FluxPipeline, StableDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from pruna import SmashConfig, smash


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, token="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").to("cuda")

        smash_config = SmashConfig()
        smash_config['compilers'] = ['flux_caching']
        smash_config['comp_flux_caching_cache_interval'] = 2
        smash_config['comp_flux_caching_start_step'] = 0
        smash_config['comp_flux_caching_compile'] = True
        smash_config['comp_flux_caching_save_model'] = False

        # # Smash the model
        self.pipe = smash(
            model=self.pipe,
            token='<your-token>',  # replace <your-token> with your actual token or set to None if you do not have one yet
            smash_config=smash_config,
        )

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        num_inference_steps: int = Input(
            description="Number of inference steps", default=4
        ),
        guidance_scale: float = Input(
            description="Guidance scale", default=7.5
        ),
        seed: int = Input(description="Seed", default=42),
        image_height: int = Input(description="Image height", default=1024),
        image_width: int = Input(description="Image width", default=1024),
        cache_interval: int = Input(description="Cache interval", default=3),
        start_step: int = Input(description="Start step", default=1),
    ) -> Path:
        """Run a single prediction on the model"""
        self.pipe.flux_cache_helper.set_params(cache_interval=cache_interval, start_step=start_step)
        image = self.pipe(
            prompt,
            height=image_height,
            width=image_width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]

        output_dir = Path(tempfile.mkdtemp())
        image_path = output_dir / "output.png"
        image.save(image_path)
        return image_path