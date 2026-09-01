"""Runtime configuration. Everything is overridable via environment variables."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- service ---
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: str = ""  # comma-separated; empty disables auth (dev only)
    cors_origins: str = "*"

    # --- ComfyUI backend ---
    comfy_host: str = "127.0.0.1"
    comfy_port: int = 8188
    comfy_root: Path = Path("/opt/ComfyUI")
    # If true the API spawns and supervises ComfyUI itself, streaming its logs
    # into our own logger. If false it expects an already-running instance.
    comfy_manage: bool = True
    comfy_extra_args: str = ""  # e.g. "--lowvram" on a <16GB card
    comfy_boot_timeout: int = 600

    # --- model files (must match what download_models.sh fetched) ---
    unet_name: str = "flux-2-klein-9b-Q8_0.gguf"
    clip_name: str = "qwen_3_8b_fp8mixed.safetensors"
    vae_name: str = "flux2-vae.safetensors"
    lora_name: str = "flux-klein-tryon-comfy.safetensors"

    # --- generation defaults (all overridable per request) ---
    # Measured on an RTX 5090 at 5 steps, one garment, three resolutions:
    #   0.50 MP (624x832)   6.5s
    #   0.75 MP (768x1024)  7.4s
    #   1.00 MP (896x1152)  9.0s
    # which is about 4s of fixed cost — VAE-encoding the two reference images,
    # setup, decode — plus ~5s per megapixel of sampling. Dropping to 0.5 MP
    # 0.75 MP is the default: it costs 13% more than 0.5 MP for 50% more
    # pixels, and saves 18% against 1.0 MP. The guardrail does not scale with
    # resolution, so in the guarded pipeline the total only moves 11% across
    # the whole range - the resolution lever matters most when generation
    # runs unguarded.
    # (An RTX 4090 measured 9.1-9.7s at 0.75 MP for the same settings.)
    default_steps: int = 5
    default_cfg: float = 1.0
    default_lora_strength: float = 0.4
    default_megapixels: float = 0.75

    # --- job handling ---
    workers: int = 1  # ComfyUI serialises on the GPU anyway
    job_ttl_seconds: int = 3600
    max_upload_bytes: int = 20 * 1024 * 1024


    # --- phases and models -------------------------------------------------
    # Two workloads, 24 GB, never resident together. See docs/ARCHITECTURE.md.
    #   brochure  32B 4-bit, ~20 GB, alone. Shown one image at a time, so the
    #             remaining ~4 GB is enough for activations.
    #   generate  ComfyUI 16.6 GB + guardrail 5.5 GB = 22.1 GB.
    brochure_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    # bf16, not quantised. It fits alone (17.5 GB of 24) and it removes
    # bitsandbytes from the path entirely - two separate 32B attempts died on
    # quantisation, one on a pre-quantised checkpoint whose FP4 state never
    # initialised, one on accelerate refusing to place it. This stage also
    # failed on *perception* before (a black lehenga read as grey, a gown read
    # as a saree), which is the worst place to add quantisation error.
    brochure_quantise: str = "none"
    guardrail_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    guardrail_quantise: str = "4bit"
    # A hard ceiling so the guardrail cannot creep into the memory ComfyUI
    # needs. Without it a long run OOMs the generator instead of the VLM.
    guardrail_max_memory_gb: float = 7.0
    guardrail_enabled: bool = True
    # Reserve must actually cover the guardrail's footprint. The 4.0 that
    # worked on a 32 GB card relied on slack that a 24 GB card does not have.
    reserve_vram_gb: float = 6.0
    start_phase: str = "generate"
    garment_dir: Path = Path("./inputs/fg")
    hf_cache_dir: str | None = None
    garment_cache_dir: Path = Path("./cache/garments")
    max_retries: int = 3

    # --- test platform auth ------------------------------------------------
    # username:bcrypt-or-plain pairs, comma separated. Empty disables login.
    auth_users: str = ""
    auth_secret: str = ""          # HMAC key for tokens; generated if empty
    auth_token_ttl: int = 43200    # 12 hours

    workflow_path: Path = Path(__file__).parent.parent / "workflows" / "tryon_api.json"
    output_dir: Path = Path("./outputs")

    @property
    def comfy_base(self) -> str:
        return f"http://{self.comfy_host}:{self.comfy_port}"

    @property
    def comfy_ws(self) -> str:
        return f"ws://{self.comfy_host}:{self.comfy_port}/ws"

    @property
    def key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
