"""
Automated test for InternVL2.5-26B integration.

Tests (in order):
  1. Model files present — output directory exists with expected IR files
  2. VLMPipeline loads — model loads on GPU.1 without error
  3. Text-only inference — model answers a simple question without an image
  4. Image inference — model describes a synthetic test image via server API
  5. Routing — Auto model with image routes to a VLM (not text LLM)
  6. Explicit model selection — sending model=internvl2.5-26b-int4-ov hits that model
  7. Adapter detection — InternVLAdapter is returned for the InternVL tokenizer

Usage:
  python3 autotest/test_internvl.py            # full suite
  python3 autotest/test_internvl.py --load-only  # tests 1-2 only (quick smoke)
  python3 autotest/test_internvl.py --api-only   # tests 4-6 only (server must be up)
"""
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import httpx

MODEL_DIR = Path("/opt/ov_server/models/internvl2.5-26b-int4-ov")
DEVICE = "GPU.1"
BASE = "http://localhost:11435"
TIMEOUT = 300.0

# Minimal 64×64 red square as base64 PNG — no external files needed
_RED_PNG_B64 = None


def _make_test_image_b64() -> str:
    """Generate a 64x64 red square PNG and return as base64 data URI."""
    global _RED_PNG_B64
    if _RED_PNG_B64:
        return _RED_PNG_B64
    try:
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (64, 64), color=(220, 30, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        _RED_PNG_B64 = f"data:image/png;base64,{b64}"
        return _RED_PNG_B64
    except ImportError:
        # Fallback: minimal valid 1x1 red PNG (hardcoded bytes)
        raw = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
            b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        _RED_PNG_B64 = "data:image/png;base64," + base64.b64encode(raw).decode()
        return _RED_PNG_B64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Test 1 — model files present
# ---------------------------------------------------------------------------

def test_files_present() -> bool:
    print("\n[1] Model files present")
    ok = True
    ok &= _check("output directory exists", MODEL_DIR.exists(), str(MODEL_DIR))
    if not MODEL_DIR.exists():
        return False
    ir_files = list(MODEL_DIR.glob("*.xml"))
    ok &= _check("at least one .xml IR file", len(ir_files) > 0,
                 f"found: {[f.name for f in ir_files[:4]]}")
    total_gb = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file()) / 1e9
    ok &= _check("output size reasonable (>5 GB)", total_gb > 5,
                 f"{total_gb:.1f} GB")
    print(f"    → {len(ir_files)} IR file(s), {total_gb:.1f} GB total")
    return ok


# ---------------------------------------------------------------------------
# Test 2 — VLMPipeline loads
# ---------------------------------------------------------------------------

def test_pipeline_loads() -> bool:
    print("\n[2] VLMPipeline loads on GPU.1")
    try:
        import openvino_genai as ov_genai
    except ImportError:
        return _check("openvino_genai importable", False, "not installed")

    if not MODEL_DIR.exists():
        return _check("model directory present", False, str(MODEL_DIR))

    t0 = time.perf_counter()
    try:
        pipe = ov_genai.VLMPipeline(str(MODEL_DIR), DEVICE)
        elapsed = time.perf_counter() - t0
        _check("VLMPipeline(model_dir, GPU.1) succeeds", True, f"{elapsed:.0f}s")
        return True
    except Exception as exc:
        _check("VLMPipeline loads", False, str(exc)[:200])
        return False


# ---------------------------------------------------------------------------
# Test 3 — text-only inference through loaded pipeline
# ---------------------------------------------------------------------------

def test_text_inference() -> bool:
    print("\n[3] Text-only inference (direct VLMPipeline)")
    try:
        import openvino_genai as ov_genai
        from transformers import AutoTokenizer
    except ImportError as e:
        return _check("imports available", False, str(e))

    if not MODEL_DIR.exists():
        return _check("model directory present", False)

    try:
        pipe = ov_genai.VLMPipeline(str(MODEL_DIR), DEVICE)
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": "Reply with exactly the word HELLO and nothing else."}],
            tokenize=False, add_generation_prompt=True,
        )
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = 10
        cfg.temperature = 0.1
        cfg.do_sample = False
        t0 = time.perf_counter()
        result = pipe.generate(prompt, generation_config=cfg)
        elapsed = time.perf_counter() - t0
        text = str(result).strip()
        ok = "hello" in text.lower() or len(text) > 0
        _check("model generates text", ok, f"{text!r} ({elapsed:.1f}s)")
        return ok
    except Exception as exc:
        _check("text inference", False, str(exc)[:200])
        return False


# ---------------------------------------------------------------------------
# Test 4 — image inference via server API
# ---------------------------------------------------------------------------

def test_image_via_api(model: str) -> bool:
    print(f"\n[4] Image inference via server API ({model})")
    img_b64 = _make_test_image_b64()
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_b64}},
                {"type": "text", "text": "What colour is this image? Reply in one word."},
            ],
        }],
        "max_tokens": 20,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        resp = _post("/v1/chat/completions", payload)
        elapsed = time.perf_counter() - t0
        content = resp["choices"][0]["message"]["content"]
        ok = isinstance(content, str) and len(content.strip()) > 0
        colour_ok = any(w in content.lower() for w in ("red", "crimson", "scarlet"))
        _check("response received", ok, f"{content[:80]!r} ({elapsed:.1f}s)")
        _check("mentions red colour", colour_ok, content[:80])
        return ok and colour_ok
    except Exception as exc:
        _check("API call succeeds", False, str(exc)[:200])
        return False


# ---------------------------------------------------------------------------
# Test 5 — routing: Auto + image → VLM (not text model)
# ---------------------------------------------------------------------------

def test_routing_auto() -> bool:
    print("\n[5] Routing: Auto model + image → VLM selected")
    img_b64 = _make_test_image_b64()
    payload = {
        "model": "Auto",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_b64}},
                {"type": "text", "text": "Describe the image briefly."},
            ],
        }],
        "max_tokens": 30,
        "stream": False,
    }
    try:
        resp = _post("/v1/chat/completions", payload)
        model_used = resp.get("model", "")
        decision = _get("/health").get("last_routing_decision") or {}
        task_class = decision.get("task_class", "")
        ok = True
        ok &= _check("task_class=vision", task_class == "vision", f"got '{task_class}'")
        ok &= _check("VLM model used", "vl" in model_used.lower() or "internvl" in model_used.lower(),
                     f"model={model_used}")
        return ok
    except Exception as exc:
        _check("routing test", False, str(exc)[:200])
        return False


# ---------------------------------------------------------------------------
# Test 6 — explicit model selection
# ---------------------------------------------------------------------------

def test_explicit_model() -> bool:
    print("\n[6] Explicit model=internvl2.5-26b-int4-ov selection")
    img_b64 = _make_test_image_b64()
    payload = {
        "model": "internvl2.5-26b-int4-ov",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_b64}},
                {"type": "text", "text": "What colour dominates this image?"},
            ],
        }],
        "max_tokens": 20,
        "stream": False,
    }
    try:
        resp = _post("/v1/chat/completions", payload)
        model_used = resp.get("model", "")
        ok = _check("model=internvl2.5-26b-int4-ov used",
                    "internvl" in model_used.lower(), f"got '{model_used}'")
        content = resp["choices"][0]["message"]["content"]
        print(f"    → answer: {content[:100]!r}")
        return ok
    except Exception as exc:
        _check("explicit model selection", False, str(exc)[:200])
        return False


# ---------------------------------------------------------------------------
# Test 7 — adapter detection
# ---------------------------------------------------------------------------

def test_adapter_detection() -> bool:
    print("\n[7] Adapter detection for InternVL tokenizer")
    try:
        import sys
        sys.path.insert(0, "/opt/ov_server")
        from transformers import AutoTokenizer
        from prompt_builder import get_adapter, InternVLAdapter

        if not MODEL_DIR.exists():
            return _check("model dir present", False)

        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        adapter = get_adapter(tok)
        ok = isinstance(adapter, InternVLAdapter)
        _check("get_adapter returns InternVLAdapter",
               ok, f"got {type(adapter).__name__}")
        if ok:
            _check("max_context_tokens=8192",
                   adapter.max_context_tokens == 8192,
                   str(adapter.max_context_tokens))
        return ok
    except Exception as exc:
        _check("adapter detection", False, str(exc)[:200])
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-only", action="store_true",
                        help="Run tests 1-2 only (files + pipeline load)")
    parser.add_argument("--api-only", action="store_true",
                        help="Run tests 4-6 only (requires server running)")
    args = parser.parse_args()

    passed = failed = 0

    def run(fn, *a) -> bool:
        nonlocal passed, failed
        ok = fn(*a)
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
        run(test_image_via_api, "internvl2.5-26b-int4-ov")
        run(test_routing_auto)
        run(test_explicit_model)
    elif args.load_only:
        run(test_files_present)
        run(test_pipeline_loads)
    else:
        run(test_files_present)
        run(test_pipeline_loads)
        run(test_text_inference)
        try:
            _get("/health")
            run(test_image_via_api, "internvl2.5-26b-int4-ov")
            run(test_routing_auto)
            run(test_explicit_model)
        except Exception:
            print("\n  (server not reachable — skipping API tests 4-6)")
        run(test_adapter_detection)

    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
