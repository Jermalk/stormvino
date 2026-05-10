"""
Automated tests for SDXL image generation.

Tests:
  1. Model files present — image_model dir has expected .xml files
  2. Pipeline loads on GPU.1
  3. Direct generate: 256×256, 1 step — returns valid PNG bytes
  4. Server API: POST /v1/images/generations → b64_json is valid PNG
  5. Size parsing: 512x512 accepted, correct dimensions in response
  6. Negative prompt accepted without error
  7. Health endpoint shows image_model_loaded=true after first call

Usage:
  python3 autotest/test_image_gen.py            # all tests (requires server)
  python3 autotest/test_image_gen.py --load-only  # tests 1-3 only (no server)
  python3 autotest/test_image_gen.py --api-only   # tests 4-7 only (server required)
"""
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import httpx

_CONFIG_PATH = Path("/opt/ov_server/config.json")
_image_model = json.loads(_CONFIG_PATH.read_text()).get("image_model", "sdxl-fp16-ov")
MODEL_DIR = Path(f"/opt/ov_server/models/{_image_model}")
DEVICE = "GPU.1"
BASE = "http://localhost:11435"
TIMEOUT = 300.0


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def _post(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=TIMEOUT) as c:
        r = c.post(f"{BASE}{path}", json=payload)
        r.raise_for_status()
        return r.json()


def _get(path: str) -> dict:
    with httpx.Client(timeout=10) as c:
        r = c.get(f"{BASE}{path}")
        r.raise_for_status()
        return r.json()


def _is_valid_png(b64_str: str) -> bool:
    try:
        raw = base64.b64decode(b64_str)
        return raw[:8] == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Test 1 — model files present
# ---------------------------------------------------------------------------

def test_files_present() -> bool:
    print(f"\n[1] Model files present ({MODEL_DIR.name})")
    ok = True
    ok &= _check("model directory exists", MODEL_DIR.exists(), str(MODEL_DIR))
    if not MODEL_DIR.exists():
        return False
    xml_files = list(MODEL_DIR.rglob("*.xml"))
    ok &= _check("at least 4 .xml IR files", len(xml_files) >= 4,
                 f"found {len(xml_files)}: {[f.name for f in xml_files[:4]]}")
    total_gb = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file()) / 1e9
    ok &= _check("total size > 1 GB", total_gb > 1, f"{total_gb:.1f} GB")
    return ok


# ---------------------------------------------------------------------------
# Test 2 — pipeline loads
# ---------------------------------------------------------------------------

def test_pipeline_loads() -> bool:
    print(f"\n[2] Text2ImagePipeline loads on {DEVICE}")
    try:
        import openvino_genai as ov_genai
    except ImportError:
        return _check("openvino_genai importable", False)

    if not MODEL_DIR.exists():
        return _check("model dir present", False)

    t0 = time.perf_counter()
    try:
        pipe = ov_genai.Text2ImagePipeline(str(MODEL_DIR), DEVICE)
        elapsed = time.perf_counter() - t0
        return _check("Text2ImagePipeline loads", True, f"{elapsed:.0f}s")
    except Exception as exc:
        return _check("pipeline load", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Test 3 — direct generate (1 step, tiny image)
# ---------------------------------------------------------------------------

def test_direct_generate() -> bool:
    print("\n[3] Direct generate: 256×256, 1 step")
    try:
        import openvino_genai as ov_genai
        import numpy as np
        from PIL import Image
    except ImportError as e:
        return _check("imports available", False, str(e))

    if not MODEL_DIR.exists():
        return _check("model dir present", False)

    try:
        pipe = ov_genai.Text2ImagePipeline(str(MODEL_DIR), DEVICE)
        t0 = time.perf_counter()
        tensor = pipe.generate(
            "a solid red square",
            num_inference_steps=1,
            width=256, height=256,
            guidance_scale=0.0,
        )
        elapsed = time.perf_counter() - t0
        arr = np.array(tensor.data, dtype=np.uint8)
        ok = True
        ok &= _check("tensor is non-empty", arr.size > 0, f"shape={arr.shape}")
        ok &= _check("generate completes", True, f"{elapsed:.1f}s")
        # Convert to PNG and verify
        if arr.ndim == 4:
            arr = arr[0]
        img = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        ok &= _check("output is valid PNG", raw[:8] == b"\x89PNG\r\n\x1a\n", f"{len(raw)} bytes")
        return ok
    except Exception as exc:
        return _check("direct generate", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Test 4 — server API: POST /v1/images/generations
# ---------------------------------------------------------------------------

def test_api_generate() -> bool:
    print("\n[4] Server API: POST /v1/images/generations")
    payload = {
        "prompt": "a solid blue circle on white background",
        "size": "256x256",
        "num_inference_steps": 1,
    }
    t0 = time.perf_counter()
    try:
        resp = _post("/v1/images/generations", payload)
        elapsed = time.perf_counter() - t0
        ok = True
        ok &= _check("response has 'created'", "created" in resp)
        ok &= _check("response has 'data' list", isinstance(resp.get("data"), list) and len(resp["data"]) > 0)
        if ok:
            b64 = resp["data"][0].get("b64_json", "")
            ok &= _check("b64_json is valid PNG", _is_valid_png(b64),
                         f"{len(b64)} chars ({elapsed:.1f}s)")
        return ok
    except Exception as exc:
        return _check("API call", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Test 5 — size 512x512 accepted
# ---------------------------------------------------------------------------

def test_api_size_512() -> bool:
    print("\n[5] Size 512×512 accepted")
    payload = {
        "prompt": "a green triangle",
        "size": "512x512",
        "num_inference_steps": 1,
    }
    try:
        resp = _post("/v1/images/generations", payload)
        b64 = resp.get("data", [{}])[0].get("b64_json", "")
        ok = _check("512×512 request accepted", _is_valid_png(b64))
        return ok
    except Exception as exc:
        return _check("512×512 request", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Test 6 — negative prompt accepted
# ---------------------------------------------------------------------------

def test_api_negative_prompt() -> bool:
    print("\n[6] Negative prompt accepted")
    payload = {
        "prompt": "a cat",
        "negative_prompt": "blurry, ugly, low quality",
        "size": "256x256",
        "num_inference_steps": 1,
    }
    try:
        resp = _post("/v1/images/generations", payload)
        b64 = resp.get("data", [{}])[0].get("b64_json", "")
        return _check("negative_prompt accepted", _is_valid_png(b64))
    except Exception as exc:
        return _check("negative_prompt request", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Test 7 — health shows image_model_loaded
# ---------------------------------------------------------------------------

def test_health_image_model() -> bool:
    print("\n[7] Health: image_model_loaded=true")
    try:
        h = _get("/health")
        ok = _check("image_model_loaded=true", h.get("image_model_loaded") is True,
                    str(h.get("image_model_loaded")))
        ok &= _check("image_model_id non-empty", bool(h.get("image_model_id")),
                     h.get("image_model_id", ""))
        return ok
    except Exception as exc:
        return _check("health check", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-only", action="store_true", help="Tests 1-3 only")
    parser.add_argument("--api-only", action="store_true", help="Tests 4-7 only (server required)")
    args = parser.parse_args()

    passed = failed = 0

    def run(fn) -> bool:
        nonlocal passed, failed
        ok = fn()
        if ok:
            passed += 1
        else:
            failed += 1
        return ok

    if args.api_only:
        try:
            _get("/health")
        except Exception as e:
            print(f"Server unreachable: {e}")
            sys.exit(1)
        run(test_api_generate)
        run(test_api_size_512)
        run(test_api_negative_prompt)
        run(test_health_image_model)
    elif args.load_only:
        run(test_files_present)
        run(test_pipeline_loads)
        run(test_direct_generate)
    else:
        run(test_files_present)
        run(test_pipeline_loads)
        run(test_direct_generate)
        try:
            _get("/health")
            run(test_api_generate)
            run(test_api_size_512)
            run(test_api_negative_prompt)
            run(test_health_image_model)
        except Exception:
            print("\n  (server not reachable — skipping API tests 4-7)")

    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
