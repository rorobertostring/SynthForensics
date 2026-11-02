import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline


ORDERED_KEYS = [
    "style",
    "shot_type",
    "short_description",
    "dense_description",
    "background",
    "lighting",
    "atmosphere",
    "camera_movement",
]

FIXED_ENV = {
    "repo_url": "https://github.com/guandeh17/Self-Forcing",
    "repo_commit": "33593df3e81fa3ec10239271dd2c100facac6de1",
    "model_id": "Self-Forcing",
    "model_url": "https://huggingface.co/gdhe17/Self-Forcing",
    "model_hf_commit": "6014fb261a5bf90e9328953b49cceb8f9654249e",
    "model_sha256": {
        "checkpoints/ode_init.pt": "b5396b8076ab3b920c9e4f4a2b52daa2c98c9983fb5e067ae5160fdf778dce21",
        "checkpoints/self_forcing_10s.pt": "c7885ec2df188ce05e63fc768aecf9a95b5672dbc64e82ea6e2a48751f67dc11",
        "checkpoints/self_forcing_dmd.pt": "a0413986d9734e02c09504e1520f5697ba6df731bb2f0f35577485e9cc8f56a3",
        "checkpoints/self_forcing_gan.pt": "4005372c17c4dfc27ceae8c951c69bc22256d0b67eede2da234b382fba1d8a2f",
        "checkpoints/self_forcing_sid.pt": "bb354a0af5775c6316a6805db8ace9b6093190d4d07f93fd83f3a88c5ad49b19",
        "checkpoints/self_forcing_sid_v2.pt": "b98050d304569c1880e46f9b05d375d8b1370882aa72f0d53b068f3f03909b40",
    },
    "base_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
    "base_model_sha256": {
        "Wan2.1_VAE.pth": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
        "diffusion_pytorch_model.safetensors": "96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f",
        "models_t5_umt5-xxl-enc-bf16.pth": "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
    },
}


def build_prompt_from_json(json_data):
    prompt_parts = []
    for key in ORDERED_KEYS:
        value = json_data.get(key, "").strip()
        if not value:
            continue
        if len(value.split()) < 4:
            prompt_parts.append(f"The {key.replace('_', ' ')} is {value.lower()}")
        else:
            prompt_parts.append(value)

    prompt = ". ".join(prompt_parts)
    while ".." in prompt:
        prompt = prompt.replace("..", ".")
    return prompt


def get_already_processed_files(output_dir):
    already_processed = set()
    for meta_file in Path(output_dir).glob("*_self-forcing_t2v.json"):
        base_name = meta_file.stem.replace("_self-forcing_t2v", "")
        if Path(output_dir).joinpath(f"{base_name}_self-forcing_t2v.mp4").exists():
            already_processed.add(base_name)
    return already_processed


def get_computed_env():
    try:
        import flash_attn
        flash_attn_version = flash_attn.__version__
    except ImportError:
        flash_attn_version = "N/A"

    key_deps = {}
    for pkg in ("diffusers", "accelerate", "transformers"):
        try:
            key_deps[pkg] = __import__(pkg).__version__
        except ImportError:
            key_deps[pkg] = "N/A"

    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__.split("+")[0],
        "flash_attn_version": flash_attn_version,
        "cuda_version": torch.version.cuda if torch.version.cuda else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "key_deps": key_deps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple videos from caption JSONs using Self-Forcing"
    )
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--captions_dir", type=str, required=True, help="Path to the captions folder")
    parser.add_argument("--output_folder", type=str, default="./output", help="Output folder for videos and metadata")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of output video frames (81 = 5s at 16fps)")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Free VRAM: {get_cuda_free_memory_gb(gpu):.2f} GB")
    low_memory = get_cuda_free_memory_gb(gpu) < 40

    torch.set_grad_enabled(False)

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    if hasattr(config, "denoising_step_list"):
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)

    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    if "generator" in state_dict:
        pipeline.generator.load_state_dict(state_dict["generator"])
    elif "generator_ema" in state_dict:
        pipeline.generator.load_state_dict(state_dict["generator_ema"])
    else:
        # flat state dict — weights directly at top level
        pipeline.generator.load_state_dict(state_dict)

    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    # Sampling run by CausalInferencePipeline: denoising steps and shift from the config, no CFG or negative prompt
    sample_steps = len(config.denoising_step_list)
    sample_shift = float(config.timestep_shift)

    captions_dir = Path(args.captions_dir)
    json_files = sorted(captions_dir.glob("*.json"))

    print("=" * 80)
    print(f"Found {len(json_files)} caption files in {captions_dir}")
    print("=" * 80)

    already_processed = get_already_processed_files(args.output_folder)
    print(f"Already processed: {len(already_processed)} files")

    json_files = [f for f in json_files if f.stem not in already_processed]
    print(f"Remaining: {len(json_files)} files to process")
    print("=" * 80)

    computed_env = get_computed_env()
    environment = {**FIXED_ENV, **computed_env}
    environment["model_sha256"] = {
        k: v for k, v in FIXED_ENV["model_sha256"].items() if Path(k).name == Path(args.checkpoint_path).name
    }

    # WAN2.1 VAE uses 4x temporal compression:
    # num_video_frames = (num_latent_frames - 1) * 4 + 1
    # For 81 video frames -> 21 latent frames
    num_latent_frames = (args.num_frames - 1) // 4 + 1

    resolutions = [(832, 480), (480, 832)]

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        prompt = build_prompt_from_json(json_data)
        seed = random.randint(0, 2**32 - 1)
        width, height = random.choice(resolutions)

        print("=" * 80)
        print(f"Processing : {json_file.name}")
        print(f"Resolution : {width}x{height}  |  Seed: {seed}")
        print(f"Prompt     : {prompt[:120]}...")
        print("=" * 80)

        filename = json_file.stem
        output_path = os.path.join(args.output_folder, f"{filename}_self-forcing_t2v.mp4")
        metadata_path = os.path.join(args.output_folder, f"{filename}_self-forcing_t2v.json")

        # Latent spatial dims: WAN uses 8x spatial downsampling
        h_latent = height // 8
        w_latent = width // 8

        torch.manual_seed(seed)
        sampled_noise = torch.randn(
            [1, num_latent_frames, 16, h_latent, w_latent],
            device=device,
            dtype=torch.bfloat16,
        )

        start_time = time.time()
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=[prompt],
            return_latents=True,
            initial_latent=None,
            low_memory=low_memory,
        )
        generation_time = round(time.time() - start_time, 1)

        video_frames = (255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()).to(torch.uint8)
        pipeline.vae.model.clear_cache()

        write_video(output_path, video_frames[0], fps=args.fps)
        print(f"Saved video    : {output_path}  ({generation_time}s)")

        metadata = {
            "generator": "Self-Forcing",
            "resolution": "480p",
            "prompt": prompt,
            "negative_prompt": None,
            "seed": seed,
            "width": width,
            "height": height,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "sample_steps": sample_steps,
            "sample_shift": sample_shift,
            "guide_scale": 1.0,
            "config": args.config_path,
            "checkpoint": args.checkpoint_path,
            "source_caption": json_file.name,
            "generation_time_seconds": generation_time,
            "environment": environment,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata : {metadata_path}")


if __name__ == "__main__":
    main()
