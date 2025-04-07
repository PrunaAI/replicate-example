import tempfile
from pathlib import Path

import torch
from cog import BasePredictor, Input, Path
from diffusers import FluxPipeline
from PIL import Image
from pruna_pro import SmashConfig, smash


def save_image(
    output_folder: Path,
    seed: int,
    index: int | str,
    image: Image.Image,
    output_format: str,
    output_quality: int,
    step: int | None = None,
) -> Path:
    """Save the image to the disk.

    Args:
        output_folder (Path)
        seed (int)
        index (int): expected to be the output number (0 to num_output- 1); optionally additional info
        image (Image.Image): the image to save
        output_format (str)
        output_quality (int)
        step (int | None, optional): denoising step, used for intermediate images only. Defaults to None.

    Raises:
        ValueError: if the output format is invalid
        ValueError: if the output quality is invalid

    Returns:
        Path: the path to the saved image
    """
    if output_format == "jpg":
        output_format = "jpeg"

    # Check the output format
    if output_format not in ["webp", "png", "jpeg"]:
        raise ValueError(f"Invalid output format: {output_format}")

    # Check the output quality
    if output_quality < 0 or output_quality > 100:
        raise ValueError(f"Invalid output quality: {output_quality}")

    output_path = (
        output_folder / f"output_{seed!s}_{index}.{output_format}"
        if step is None
        else output_folder
        / f"output_{seed!s}_{index}_intermediate_{step}.{output_format}"
    )
    if output_format != "png":
        image.save(str(output_path.resolve()), quality=output_quality, optimize=True)
    else:
        image.save(str(output_path.resolve()))
    return Path(output_path)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        model_path = "black-forest-labs/FLUX.1-dev"
        # Load base pipeline (will be txt2img)
        base_pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        print("Loading pipeline")

        # Configure smashing
        smash_config = SmashConfig()
        smash_config["cacher"] = "taylor_auto"
        smash_config["compiler"] = "torch_compile"
        smash_config._prepare_saving = False
        smash_token = "<your-token>"

        # Smash the model and store it
        print("Smashing txt2img pipeline...")
        self.smashed_txt2img_pipe = smash(
            model=base_pipe,
            token=smash_token,
            smash_config=smash_config,
        )
        print("Setup complete.")

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        speed_mode: str = Input(
            description="Speed optimization level",
            default="Juiced 🔥 (default)",
            choices=[
                "Lightly Juiced 🍊 (more consistent)",
                "Juiced 🔥 (default)",
                "Extra Juiced 🔥 (more speed)",
            ],
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps", default=28
        ),
        guidance: float = Input(description="Guidance scale", default=7.5),
        seed: int = Input(description="Seed", default=-1),
        aspect_ratio: str = Input(
            description="Aspect ratio of the output image",
            default="1:1",
            choices=[
                "1:1",
                "16:9",
                "21:9",
                "3:2",
                "2:3",
                "4:5",
                "5:4",
                "3:4",
                "4:3",
                "9:16",
                "9:21",
            ],
        ),
        image_size: int = Input(
            description="Base image size (longest side)", default=1024
        ),
        output_format: str = Input(
            description="Output format", default="png", choices=["png", "jpg", "webp"]
        ),
        output_quality: int = Input(
            description="Output quality (for jpg and webp)", default=80, ge=1, le=100
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # Calculate dimensions based on aspect ratio
        width, height = map(int, aspect_ratio.split(":"))
        ratio = width / height

        if ratio >= 1:
            # Width is greater or equal to height
            image_width = image_size
            image_height = int(image_size / ratio)
        else:
            # Height is greater than width
            image_height = image_size
            image_width = int(image_size * ratio)

        # Arguments for txt2img pipeline
        kwargs = {
            "prompt": prompt,
            "height": image_height,
            "width": image_width,
            "guidance_scale": guidance,
            "num_inference_steps": num_inference_steps,
            "generator": torch.Generator("cpu").manual_seed(seed)
            if seed != -1
            else None,
        }

        # Use txt2img pipeline
        print("Using txt2img pipeline")
        pipe_to_call = self.smashed_txt2img_pipe

        # Configure the cache helper for the pipeline
        if hasattr(pipe_to_call, "cache_helper"):
            pipe_to_call.cache_helper.disable()
            pipe_to_call.cache_helper.enable()
            if speed_mode == "Lightly Juiced 🍊 (more consistent)":
                print("Setting cache speed factor: 0.4")
                pipe_to_call.cache_helper.set_params(
                    speed_factor=0.5 if num_inference_steps > 20 else 0.6,
                )
            elif speed_mode == "Extra Juiced 🔥 (more speed)":
                print("Setting cache speed factor: 0.2")
                pipe_to_call.cache_helper.set_params(
                    speed_factor=0.3 if num_inference_steps > 20 else 0.4,
                )
            elif speed_mode == "Juiced 🔥 (default)":
                print("Setting cache speed factor: 0.5")
                pipe_to_call.cache_helper.set_params(
                    speed_factor=0.4 if num_inference_steps > 20 else 0.5,
                )
        else:
            print("Warning: Selected pipeline does not have cache_helper.")

        # Run the prediction
        print(f"Running prediction with args: {list(kwargs.keys())}")
        output_image = pipe_to_call(**kwargs).images[0]

        # Create output directory and save the image
        output_dir = Path(tempfile.mkdtemp())
        image_path = save_image(
            output_folder=output_dir,
            seed=seed,
            index=0,  # First (and only) output
            image=output_image,  # Pass the generated PIL image
            output_format=output_format,
            output_quality=output_quality,
        )

        return image_path
