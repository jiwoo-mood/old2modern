import os
import math
import random
import argparse
import torch

if not hasattr(torch.nn, "RMSNorm"):
    class _RMSNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape))
            else:
                self.register_parameter("weight", None)

        def forward(self, x):
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            x_norm = x / rms
            if self.weight is None:
                return x_norm
            return x_norm * self.weight

    torch.nn.RMSNorm = _RMSNorm

_sdpa = torch.nn.functional.scaled_dot_product_attention
_sig_text = getattr(_sdpa, "__text_signature__", "") or ""
if "enable_gqa" not in _sig_text and not getattr(_sdpa, "_gqa_compat", False):
    _orig_sdpa = _sdpa

    def _sdpa_compat(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return _orig_sdpa(*args, **kwargs)

    _sdpa_compat._gqa_compat = True
    torch.nn.functional.scaled_dot_product_attention = _sdpa_compat

from pathlib import Path
from PIL import Image
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from torchvision.transforms import CenterCrop, RandomCrop

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

def apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    """Apply gamma correction; gamma < 1 brightens, gamma > 1 darkens."""
    if gamma == 1.0:
        return image
    if gamma <= 0:
        raise ValueError("Gamma must be positive")

    lut = [min(255, max(0, int(pow(i / 255.0, gamma) * 255 + 0.5))) for i in range(256)]
    return image.point(lut * len(image.getbands()))


def main():
    parser = argparse.ArgumentParser(description="Run inference with FLUX.1-Kontext-dev")
    
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt", nargs='+', required=True)
    parser.add_argument("--cols", type=int, default=3)
    
    parser.add_argument("--output_save", type=str, default="individual", choices=["individual", "grid", "both"])
    
    parser.add_argument("--num_samples", type=int, default=10000000)
    
    parser.add_argument("--model_name",   type=str, default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--cache_dir", type=str, default="/workspace/.cache")
    parser.add_argument("--local_files_only", action="store_true", help="Load the pipeline strictly from the local cache")
    parser.add_argument("--guidance_scale", type=float, default=2.5)#건들지 말것. 
    parser.add_argument("--num_inference_steps", type=int, default=30) #건들지 말것. 
    parser.add_argument("--seed", type=int, default=2025) #시드 고정 건들지 말것. 
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt for inference")
    parser.add_argument("--crop", action="store_true", help="Crop images to the preferred resolution before processing")#건들지 말것. 
    parser.add_argument("--crop_size", type=tuple, default=(1024, 1024), help="Size to crop images to if --crop is enabled")#건들지 말것. 
    
    args = parser.parse_args()
    
    # Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    
    # TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Generator
    generator = torch.Generator(device=device)
    
    # Pipe
    pipe = DiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only
    ).to(device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    image_paths = list(Path(args.input_dir).glob("*.png")) + list(Path(args.input_dir).glob("*.jpg"))
    
    accelerator.print(f"Found {len(image_paths)} images in {args.input_dir}")
    
    if args.num_samples == "None":
        args.num_samples = None
    
    if args.num_samples is not None:
        random.shuffle(image_paths)
        num_samples = min(len(image_paths), args.num_samples)
        image_paths = image_paths[:num_samples]
    
    image_paths = sorted(image_paths, key=lambda p: p.stem)  # sort by filename stem
    distributed_paths = image_paths[rank :: world_size]  # distribute images across processes
    
    accelerator.print(f"[rank {rank}] Processing {len(distributed_paths)} images")
     
    prompt_count = len(args.prompt)

    for image_path in distributed_paths:
        # Skip frames that have already been generated for all requested outputs
        already_processed = False
        if args.output_save in ["individual", "both"]:
            for idx in range(1, prompt_count + 1):
                prompt_dir = Path(args.output_dir) / f"prompt_{idx}"
                prompt_file = prompt_dir / f"{image_path.stem}.png"
                if not prompt_file.exists():
                    already_processed = False
                    break
        if already_processed and args.output_save in ["grid", "both"]:
            grid_file = Path(args.output_dir) / "grid" / f"{image_path.stem}_grid.png"
            original_file = Path(args.output_dir) / "original" / f"{image_path.stem}.png"
            if not grid_file.exists() or not original_file.exists():
                already_processed = False

        if already_processed:
            accelerator.print(f"[{image_path.name}] Outputs already exist, skipping.")
            continue

        image = load_image(str(image_path))
        image = image.convert("RGB")

        # Crop if specified
        if args.crop:
            crop_W, crop_H = args.crop_size
            image = CenterCrop((crop_H, crop_W))(image)
            resized_image = image
            target_W, target_H = crop_W, crop_H
            # for output size
            W, H = crop_W, crop_H
        
        # Auto resize to preferred resolution
        else:
            W, H = image.size
            target_W, target_H = min(PREFERRED_KONTEXT_RESOLUTIONS, key=lambda r: abs(r[0]/r[1] - W/H))
            resized_image = image.resize((target_W, target_H), Image.LANCZOS)
            # for output size
            W, H = image.size
        
        restored_images = []
        
        lift_gamma = 1.0
        conditioned_image = resized_image

        # Run inference
        for i, prompt_text in enumerate(args.prompt):
            generator.manual_seed(args.seed)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = pipe(
                    image=conditioned_image,
                    prompt=prompt_text,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    width=target_W,
                    height=target_H,
                    generator=generator,
                    negative_prompt=args.negative_prompt
                )
        
            output_image = output.images[0]
            if lift_gamma != 1.0:
                output_image = apply_gamma(output_image, 1.0 / lift_gamma)
            
            # Resize output image to match original size or cropped size
            if output_image.size != (W, H):
                print(f"Resizing output image from {output_image.size} to {(W, H)}")
                restored = output_image.resize((W, H), Image.LANCZOS)
            else:
                restored = output_image
            
            restored_images.append(restored)

        if args.output_save in ["individual", "both"]:
            for i, restored in enumerate(restored_images, start=1):
                prompt_output_dir = Path(args.output_dir) / f"prompt_{i}"
                os.makedirs(prompt_output_dir, exist_ok=True)

                individual_filename = prompt_output_dir / f"{image_path.stem}.png"
                restored.save(individual_filename)

                accelerator.print(f"Saved individual image to {individual_filename}")
        
        # Save original image or cropped image
        if args.output_save in ["grid", "both"]:
            original_output_dir = Path(args.output_dir) / "original"
            os.makedirs(original_output_dir, exist_ok=True)
            
            original_filename = original_output_dir / f"{image_path.stem}.png"
            image.save(original_filename)
            
            accelerator.print(f"Saved original image to {original_filename}")
        
        # Save grid image if requested
        if args.output_save in ["grid", "both"]:
            all_images = [image] + restored_images
            num_images = len(all_images)
            cols = min(args.cols, num_images)
            rows = math.ceil(num_images / cols)
        
            grid_image = Image.new("RGB", (W * cols, H * rows))
        
            for i, img in enumerate(all_images):
                grid_x = (i % cols) * W
                grid_y = (i // cols) * H
                grid_image.paste(img, (grid_x, grid_y))
        
            # Save the grid image
            grid_output_dir = Path(args.output_dir) / "grid"
            os.makedirs(grid_output_dir, exist_ok=True)
            
            grid_filename = grid_output_dir / f"{image_path.stem}_grid.png"
            grid_image.save(grid_filename)

            accelerator.print(f"Processed {image_path.name} and saved to {grid_filename}")
            
            del grid_image
        
        del image, resized_image, restored_images
    
    # Save prompts to a text file for reference
    if rank == 0:
        prompt_filename = Path(args.output_dir) / "prompts.txt"
        with open(prompt_filename, "w") as f:
            f.write("This file maps prompt indices to the full prompt text used for generation.\n")
            f.write(f"The subdirectories (e.g., 'prompt_1') correspond to these indices.\n\n")
            if args.negative_prompt:
                f.write(f"--- Negative Prompt ---\n{args.negative_prompt}\n\n")
            for i, p in enumerate(args.prompt, 1):
                f.write(f"--- Prompt {i} ---\n{p}\n\n")
        for i, p in enumerate(args.prompt, 1):
            prompt_dir = Path(args.output_dir) / f"prompt_{i}"
            os.makedirs(prompt_dir, exist_ok=True)
            with open(prompt_dir / "prompt.txt", "w") as f:
                f.write(p)
                if args.negative_prompt:
                    f.write(f"\n\n--- Negative Prompt ---\n{args.negative_prompt}\n")

    # Cleanup
    accelerator.end_training()
    

if __name__ == "__main__":
    main()
