"""
Image generation pipeline wrapper.

Wraps ov_genai.Text2ImagePipeline (SDXL, SD, LCM) for use in ov_server.py.
Module-level state: _image_pipe, _image_lock.
Never import from ov_server.py.
"""
import asyncio
import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("ov_server")

_image_pipe = None
_image_model_id: str = ""
_image_lock = asyncio.Lock()


def is_loaded() -> bool:
    return _image_pipe is not None


def loaded_model_id() -> str:
    return _image_model_id


def _parse_size(size: str) -> tuple[int, int]:
    """Parse 'WxH' or 'W×H' string into (width, height). Default 1024×1024."""
    try:
        sep = "x" if "x" in size.lower() else "×"
        parts = size.lower().replace("×", "x").split("x")
        w, h = int(parts[0]), int(parts[1])
        # Round to nearest multiple of 8 (SDXL latent requirement)
        w = max(256, (w // 8) * 8)
        h = max(256, (h // 8) * 8)
        return w, h
    except Exception:
        return 1024, 1024


def tensor_to_png_b64(tensor) -> str:
    """Convert ov_genai image Tensor → PNG bytes → base64 string."""
    from PIL import Image as PILImage
    arr = np.array(tensor.data, dtype=np.uint8)
    # ov_genai returns shape (N, H, W, C)
    if arr.ndim == 4:
        arr = arr[0]
    img = PILImage.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def get_image_pipeline(model_dir: str, device: str):
    """Return loaded Text2ImagePipeline, loading it on first call."""
    global _image_pipe, _image_model_id
    async with _image_lock:
        if _image_pipe is not None:
            return _image_pipe
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        log.info(f"Loading image pipeline from {model_dir} on {device}...")
        try:
            import openvino_genai as ov_genai
            _image_pipe = await loop.run_in_executor(
                None,
                lambda: ov_genai.Text2ImagePipeline(model_dir, device)
            )
            _image_model_id = Path(model_dir).name
            log.info(f"Image pipeline loaded in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:
            log.error(f"Failed to load image pipeline: {exc}")
            raise
        return _image_pipe


async def generate_images(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 20,
    seed: Optional[int] = None,
    num_images: int = 1,
    model_dir: str = "",
    device: str = "GPU.1",
) -> list[str]:
    """Generate images; return list of base64-encoded PNGs (one per image)."""
    import openvino_genai as ov_genai
    pipe = await get_image_pipeline(model_dir, device)
    loop = asyncio.get_running_loop()

    def _run() -> list[str]:
        kwargs: dict = {
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": num_images,
        }
        if seed is not None:
            kwargs["generator"] = ov_genai.CppStdGenerator(seed)
        t0 = time.perf_counter()
        tensor = pipe.generate(prompt, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info(f"Image generation: {num_images} image(s) {width}×{height} "
                 f"{num_inference_steps} steps in {elapsed:.1f}s")
        arr = np.array(tensor.data, dtype=np.uint8)
        if arr.ndim == 3:
            arr = arr[np.newaxis]  # (1, H, W, C)
        results = []
        from PIL import Image as PILImage
        for i in range(arr.shape[0]):
            img = PILImage.fromarray(arr[i], mode="RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            results.append(base64.b64encode(buf.getvalue()).decode())
        return results

    return await loop.run_in_executor(None, _run)
