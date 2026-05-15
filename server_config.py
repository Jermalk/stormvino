"""
Owns config loading, model discovery, startup constants, and resolved model helpers.
Never import from ov_server.py, model_manager.py, router.py, or catalogue.py.
Imports: standard library only + openvino_genai (SchedulerConfig).
To add a new config key: add default to _load_config(), add to _KNOWN_CONFIG_KEYS.
"""
import json
import logging
import math
import subprocess
from pathlib import Path

import openvino_genai as ov_genai

log = logging.getLogger("ov_server")

# ---------------------------------------------------------------------------
# Config — loaded from config.json next to this script, falls back to
# defaults so the server starts even without a config file.
# ---------------------------------------------------------------------------
_CONFIG_FILE = Path(__file__).parent / "config.json"


def _load_config() -> dict:
    defaults: dict = {
        # ── hardware ────────────────────────────────────────────────────────
        "models_dir":             str(Path(__file__).parent / "models"),
        "device":                 "AUTO",
        "ov_cache_dir":           "/tmp/ov_cache_b60",
        "embedding_model":        "",
        "vision_model":           "",
        "model_aliases":          {},
        "max_loaded_models":      2,
        "kv_cache_size_gb":       8,
        "vram_headroom_gb":       1.5,
        "max_ram_percent":        75.0,
        "max_new_tokens_default": 2048,
        "vlm_max_image_turns":    1,
        "vlm_max_image_side_px":  1280,
        "enable_prefix_caching":  True,
        "max_num_batched_tokens":  4096,
        # ── routing control ─────────────────────────────────────────────────
        "provider_scope":  "local",
        "active_profile":  "fast",
        "providers":       {},
        # ── assessor ────────────────────────────────────────────────────────
        "assessor": {
            "model":            "",
            "kv_cache_size_gb": 2,
        },
        # ── routing pipeline ────────────────────────────────────────────────
        "router": {
            "embedding_threshold": 0.72,
            "long_context_tokens": 4000,
            "keywords":            {"web_search": []},
        },
        # ── behavioral profiles ─────────────────────────────────────────────
        "profiles": {
            "fast": {
                "thinking":         False,
                "max_new_tokens":   512,
                "model_preference": "fastest",
                "use_assessor":     False,
            },
            "precise": {
                "thinking":         True,
                "max_new_tokens":   4096,
                "model_preference": "balanced",
                "use_assessor":     True,
            },
            "laborious": {
                "thinking":         True,
                "max_new_tokens":   16384,
                "model_preference": "best",
                "use_assessor":     True,
            },
        },
        # ── task classes ────────────────────────────────────────────────────
        "task_classes": {
            "vision":     {"description": "Image understanding", "models": []},
            "web_search": {"description": "Web search or live information", "models": []},
            "document":   {"description": "Long document analysis", "models": []},
            "code":       {"description": "Code writing and debugging", "models": []},
            "general":    {"description": "General conversation", "models": []},
        },
        # ── legacy compat (removed from config.json; kept in defaults so
        #    existing _pick() / MAX_NEW_TOKENS_AGENT references still work
        #    until Step 2.4 migrates them to routing) ──────────────────────
        "default_model":      "",
        "agent_model":        "",
        "max_new_tokens_agent": 200,
        "routing":            {"default": "local", "model_map": {}, "backends": {}},
    }
    if _CONFIG_FILE.exists():
        try:
            with _CONFIG_FILE.open() as f:
                overrides = json.load(f)
            defaults.update(overrides)
            log.info(f"Config loaded from {_CONFIG_FILE}")
        except Exception as e:
            log.warning(f"Failed to read {_CONFIG_FILE}: {e} — using defaults")
    else:
        log.warning(f"No config.json found at {_CONFIG_FILE} — using defaults")
    return defaults


_KNOWN_CONFIG_KEYS: frozenset[str] = frozenset({
    # hardware
    "models_dir", "device", "ov_cache_dir", "embedding_model", "embedding_device", "vision_model", "vision_device",
    "model_aliases", "max_loaded_models", "kv_cache_size_gb", "model_kv_overrides", "vram_headroom_gb",
    "max_ram_percent", "max_new_tokens_default", "vlm_max_image_turns",
    "vlm_max_image_side_px", "enable_prefix_caching", "max_num_batched_tokens",
    # routing
    "provider_scope", "active_profile", "providers", "assessor", "router",
    "profiles", "task_classes", "postgres_dsn",
    # legacy compat — tolerated without warning until Step 2.4
    "default_model", "agent_model", "max_new_tokens_agent", "routing",
    # image generation
    "image_model", "image_device", "image_num_steps",
    # stt
    "stt_model", "stt_device",
    # tts
    "tts_model_dir", "tts_voice", "tts_voice_pl",
    # news
    "news",
    # plugins
    "plugins",
    # safety
    "blocked_models",
    # timing
    "inference_timeout_sec", "usd_to_pln",
})


def _validate_config(cfg: dict) -> None:
    """Log a warning for every unrecognised top-level config key. Never raises."""
    for key in cfg:
        if key not in _KNOWN_CONFIG_KEYS:
            log.warning(f"[config] Unrecognised key '{key}' — ignored")


_cfg = _load_config()
_validate_config(_cfg)

# ---------------------------------------------------------------------------
# Version info — read once at startup; never raises.
# ---------------------------------------------------------------------------
SERVER_VERSION = "0.9.0"


def _read_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


_GIT_COMMIT = _read_git_commit()
log.info(f"Server version {SERVER_VERSION} commit {_GIT_COMMIT}")

# ---------------------------------------------------------------------------
# Model discovery — scans models_dir for valid OpenVINO LLM directories.
# A directory is an LLM if it contains openvino_model.xml AND
# openvino_detokenizer.xml (distinguishes LLMs from embedding models).
# ---------------------------------------------------------------------------
def _discover_models(models_dir: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    if not models_dir.exists():
        log.warning(f"Models directory {models_dir} does not exist")
        return found
    for d in sorted(models_dir.iterdir()):
        if (d.is_dir()
                and (d / "openvino_model.xml").exists()
                and (d / "generation_config.json").exists()):
            found[d.name] = str(d)
    log.info(f"Discovered {len(found)} LLM model(s): {list(found)}")
    return found


def _discover_vlm_models(models_dir: Path) -> dict[str, str]:
    """VLM directories are distinguished by openvino_language_model.xml (split architecture)."""
    found: dict[str, str] = {}
    if not models_dir.exists():
        return found
    for d in sorted(models_dir.iterdir()):
        if d.is_dir() and (d / "openvino_language_model.xml").exists():
            found[d.name] = str(d)
    log.info(f"Discovered {len(found)} VLM model(s): {list(found)}")
    return found


MODELS_DIR = Path(_cfg["models_dir"])
DEVICE     = _cfg["device"]
CONFIG     = {
    "PERFORMANCE_HINT":                "LATENCY",
    "CACHE_DIR":                       _cfg["ov_cache_dir"],
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
}


def _detect_family_max_context(model_dir: Path) -> int:
    """Read tokenizer_config.json to detect family → return max_context_tokens."""
    tok_cfg_path = model_dir / "tokenizer_config.json"
    if not tok_cfg_path.exists():
        return 32_768
    try:
        with tok_cfg_path.open() as f:
            tok_cfg = json.load(f)
        chat_tmpl = tok_cfg.get("chat_template", "") or ""
        if "[SYSTEM_PROMPT]" in chat_tmpl:
            return 32_768  # MistralAdapter
        special = str(tok_cfg.get("additional_special_tokens", []))
        if "<IMG_CONTEXT>" in special:
            return 8_192   # InternVLAdapter
        return 32_768      # DefaultAdapter
    except Exception:
        return 32_768


def compute_kv_cache_gb(
    model_dir: Path,
    max_context_tokens: int | None = None,
    headroom: float = 1.25,
    floor_gb: float = 1.0,
) -> int:
    """Compute KV cache size (GB) from model architecture.

    Formula: num_layers × num_kv_heads × head_dim × 2 (K+V) × 2 bytes (FP16) × seq_len
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return int(_cfg.get("kv_cache_size_gb", 8))
    try:
        with cfg_path.open() as f:
            mcfg = json.load(f)
        num_layers    = int(mcfg.get("num_hidden_layers", 32))
        num_attn      = int(mcfg.get("num_attention_heads", 32))
        num_kv_heads  = int(mcfg.get("num_key_value_heads", num_attn))
        hidden_size   = int(mcfg.get("hidden_size", 4096))
        head_dim      = int(mcfg.get("head_dim", hidden_size // num_attn))
        if max_context_tokens is None:
            max_context_tokens = _detect_family_max_context(model_dir)
        kv_bytes = num_layers * num_kv_heads * head_dim * 2 * 2 * max_context_tokens
        kv_gb_raw = kv_bytes / 1e9
        result = math.ceil(kv_gb_raw * headroom)
        computed = max(result, math.ceil(floor_gb))
        log.debug(
            f"KV cache {model_dir.name}: {num_layers}L×{num_kv_heads}KVh×{head_dim}d "
            f"ctx={max_context_tokens} → {kv_gb_raw:.2f}GB×{headroom} = {computed}GB"
        )
        return computed
    except Exception as e:
        log.warning(f"KV cache compute failed for {model_dir.name}: {e} — using global default")
        return int(_cfg.get("kv_cache_size_gb", 8))


def _model_kv_gb(model_id: str) -> int:
    """Return KV cache size (GB) for model_id.

    Priority: model_kv_overrides > architecture formula > global default.
    """
    overrides = _cfg.get("model_kv_overrides", {})
    if model_id in overrides:
        return int(overrides[model_id])
    model_dir = MODELS_DIR / model_id
    if model_dir.exists():
        return compute_kv_cache_gb(model_dir)
    return int(_cfg.get("kv_cache_size_gb", 8))


def get_scheduler_config(kv_override: int | None = None) -> ov_genai.SchedulerConfig:
    sched = ov_genai.SchedulerConfig()
    sched.cache_size = kv_override if kv_override is not None else _cfg.get("kv_cache_size_gb", 8)
    sched.enable_prefix_caching = _cfg.get("enable_prefix_caching", True)
    sched.max_num_batched_tokens = _cfg.get("max_num_batched_tokens", 4096)
    return sched


MAX_RAM_PERCENT        = _cfg["max_ram_percent"]
MAX_NEW_TOKENS_DEFAULT = _cfg["max_new_tokens_default"]
MAX_NEW_TOKENS_AGENT   = _cfg["max_new_tokens_agent"]
VRAM_HEADROOM_GB       = _cfg["vram_headroom_gb"]
MODEL_ALIASES: dict[str, str] = _cfg["model_aliases"]
VLM_MAX_IMAGE_TURNS:   int = int(_cfg["vlm_max_image_turns"])
VLM_MAX_IMAGE_SIDE_PX: int = int(_cfg["vlm_max_image_side_px"])

AVAILABLE_MODELS     = _discover_models(MODELS_DIR)
AVAILABLE_VLM_MODELS = _discover_vlm_models(MODELS_DIR)

_vision_model_cfg = _cfg.get("vision_model", "")
if _vision_model_cfg and _vision_model_cfg not in AVAILABLE_VLM_MODELS:
    log.warning(f"Config vision_model='{_vision_model_cfg}' not found — known VLMs: {list(AVAILABLE_VLM_MODELS)}")
VISION_MODEL: str = _vision_model_cfg if _vision_model_cfg in AVAILABLE_VLM_MODELS else (
    next(iter(AVAILABLE_VLM_MODELS), "")
)

# Resolve default/agent model — use config value if present and valid,
# otherwise fall back to first / smallest discovered model.
def _pick(key: str, fallback_index: int) -> str:
    name = _cfg.get(key, "")
    if name and name in AVAILABLE_MODELS:
        return name
    if AVAILABLE_MODELS:
        picked = list(AVAILABLE_MODELS)[min(fallback_index, len(AVAILABLE_MODELS) - 1)]
        log.warning(f"Config '{key}={name}' not found — using '{picked}'")
        return picked
    return ""


_cfg["_resolved_default_model"] = _pick("default_model", -1)
_cfg["_resolved_agent_model"]   = _pick("agent_model",    0)
ROUTING_TRIGGER_MODELS: frozenset[str] = frozenset({"auto", "Auto", ""})


def get_default_model() -> str:
    return _cfg.get("_resolved_default_model", "")


def get_agent_model() -> str:
    return _cfg.get("_resolved_agent_model", "")


# Embedding model — not auto-discovered (loaded via different code path)
_emb_name            = _cfg.get("embedding_model", "")
EMBEDDING_MODEL_ID   = _emb_name
EMBEDDING_MODEL_PATH = str(MODELS_DIR / _emb_name) if _emb_name else ""
