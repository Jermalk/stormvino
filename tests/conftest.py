"""
Stub heavy GPU/ML dependencies before ov_server is imported.
This lets the test suite run on any machine without OpenVINO or a GPU.
"""
import sys
import types
from unittest.mock import MagicMock

# ── openvino_genai ──────────────────────────────────────────────────────────
# AsyncTokenStreamer inherits from StreamerBase, so it must be a real class.

class _FakeStreamerBase:
    def __init__(self):
        pass


class _FakeStreamingStatus:
    RUNNING = 0


_ov_genai = types.ModuleType("openvino_genai")
_ov_genai.StreamerBase = _FakeStreamerBase
_ov_genai.StreamingStatus = _FakeStreamingStatus()
_ov_genai.LLMPipeline = MagicMock()
_ov_genai.VLMPipeline = MagicMock()
_ov_genai.GenerationConfig = MagicMock()
_ov_genai.SchedulerConfig = MagicMock()
_ov_genai.Tokenizer = MagicMock()
sys.modules["openvino_genai"] = _ov_genai

# ── openvino ────────────────────────────────────────────────────────────────
# Used inside _init_vram() which is called at module load.
# Raise on get_property so _TOTAL_VRAM_GB stays None (soft cap disabled).

_ov = types.ModuleType("openvino")
_core_mock = MagicMock()
_core_mock.get_property.side_effect = RuntimeError("no GPU in test env")
_ov.Core = MagicMock(return_value=_core_mock)
_ov.Tensor = MagicMock()
sys.modules["openvino"] = _ov

# ── transformers ─────────────────────────────────────────────────────────────
# System-installed version has a huggingface-hub version conflict.
# We only need AutoProcessor and AutoTokenizer stubs.

_transformers = types.ModuleType("transformers")
_transformers.AutoProcessor = MagicMock()
_transformers.AutoTokenizer = MagicMock()
sys.modules["transformers"] = _transformers

# ── optimum.intel ────────────────────────────────────────────────────────────

_optimum = types.ModuleType("optimum")
_optimum_intel = types.ModuleType("optimum.intel")
_optimum_intel.OVModelForFeatureExtraction = MagicMock()
sys.modules["optimum"] = _optimum
sys.modules["optimum.intel"] = _optimum_intel
