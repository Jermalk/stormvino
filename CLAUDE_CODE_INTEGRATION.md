# Claude Code Integration Guide — ov_server

This guide covers connecting Claude Code (the Anthropic CLI) to your local OpenVINO
inference server so that Claude Code sessions run against your own hardware instead of
(or in addition to) Anthropic's cloud.

---

## Architecture overview

```
Claude Code (CLI)
      │  ANTHROPIC_BASE_URL=http://EnvyStorm:11435
      │  ANTHROPIC_API_KEY=<your-token-or-anything>
      ▼
ov_server  (/v1/messages)
      │
      ├─ local ──► LocalBackend ──► qwen3-14b-int4-ov  (GPU.1)
      ├─ ovh   ──► OpenAICompatBackend ──► OVH Qwen3-32B
      └─ uncle-a ► AnthropicBackend ──► api.anthropic.com  [Step 10 — not yet live]
```

The server exposes `/v1/messages` (Anthropic protocol). Claude Code sends requests
there as if talking to `api.anthropic.com`. The server maps incoming model names to
local models, remote backends, or passes them through.

---

## 1. Basic connection

### 1a. Set environment variables

Add to `~/.bashrc` (or the shell profile Claude Code inherits):

```bash
export ANTHROPIC_BASE_URL=http://localhost:11435   # or http://EnvyStorm:11435 from LAN
export ANTHROPIC_API_KEY=anything                  # required by the SDK; value ignored
                                                   # unless OV_SERVER_API_KEY is set
```

If you set `OV_SERVER_API_KEY` on the server, the same value must go here:

```bash
export ANTHROPIC_API_KEY=<your-OV_SERVER_API_KEY-value>
```

### 1b. Launch Claude Code

```bash
claude
```

Claude Code will connect to your server. All `/v1/messages` calls are served locally.

### 1c. Verify

```bash
# From Claude Code terminal or any shell:
curl -s http://localhost:11435/health | python3 -m json.tool | grep -E "status|loaded_models"
```

---

## 2. Model name mapping (current behaviour)

Claude Code sends Anthropic model names (`claude-sonnet-4-6`, `claude-opus-4-7`, …).
`config.json` maps these to local models via `model_aliases` and routes them to
backends via `routing.model_map`.

Current `config.json` mappings:

| Claude Code requests | Server receives | Backend | Inference |
|---|---|---|---|
| `claude-haiku-4-5` | `qwen3-8b-int4-ov` | local | GPU.1 |
| `claude-haiku-4-5-20251001` | `qwen3-8b-int4-ov` | local | GPU.1 |
| `claude-sonnet-4-6` | `qwen3-14b-int4-ov` | local | GPU.1 |
| `claude-opus-4-7` | routed to `ovh` | OVH backend | Qwen3-32B |

To change any mapping, edit `config.json` and restart the service:

```bash
sudo systemctl restart ov-server
```

---

## 3. Hashtag routing (proposed — requires hook + one server patch)

Type a routing tag anywhere in your Claude Code prompt to override the backend for
that request. The tag is stripped before the model sees it.

| Tag | Backend | Fallback |
|---|---|---|
| `#use-local-box` | local (qwen3-14b) | — |
| `#use-ovh` | OVH Qwen3-32B | local |
| `#use-uncle-a` | Anthropic API *(Step 10)* | local |

### 3a. Server patch — routing override file

Add the following block at the top of `_pick_backend()` in `ov_server.py`
(before the existing `routing` dict lookup):

```python
def _pick_backend(model: str) -> Backend:
    # Honour a short-lived routing override written by the Claude Code hook.
    _override_path = Path("/tmp/ov_routing_override.json")
    if _override_path.exists():
        try:
            data = json.loads(_override_path.read_text())
            if data.get("expires", 0) > time.time():
                name     = data.get("backend", "local")
                fallback = data.get("fallback", "local")
                backend  = _backends.get(name) or _backends.get(fallback) or _backends["local"]
                log.info(f"Routing override active: backend='{name}'")
                return backend
        except Exception as exc:
            log.debug(f"Routing override read error (ignored): {exc}")

    # ... existing routing logic unchanged below ...
```

### 3b. Hook script

Create `~/.claude/hooks/route-selector.sh`:

```bash
#!/usr/bin/env bash
# Claude Code UserPromptSubmit hook — detects routing hashtags and writes
# a short-lived override for ov_server._pick_backend().

INPUT=$(cat)
PROMPT=$(python3 -c "import sys,json; print(json.load(sys.stdin).get('prompt',''))" <<< "$INPUT" 2>/dev/null || echo "")

# TTL: 5 minutes — covers a multi-turn reasoning session
EXPIRES=$(python3 -c "import time; print(int(time.time()+300))")
OVERRIDE=/tmp/ov_routing_override.json

if   echo "$PROMPT" | grep -qF "#use-local-box"; then
    printf '{"backend":"local","expires":%s}\n' "$EXPIRES" > "$OVERRIDE"
elif echo "$PROMPT" | grep -qF "#use-ovh"; then
    printf '{"backend":"ovh","fallback":"local","expires":%s}\n' "$EXPIRES" > "$OVERRIDE"
elif echo "$PROMPT" | grep -qF "#use-uncle-a"; then
    printf '{"backend":"anthropic","fallback":"local","expires":%s}\n' "$EXPIRES" > "$OVERRIDE"
fi
# Always pass through — exit 0 means "allow, do not modify prompt"
```

```bash
chmod +x ~/.claude/hooks/route-selector.sh
```

### 3c. Register the hook in Claude Code settings

`~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "/home/jerzy/.claude/hooks/route-selector.sh"
          }
        ]
      }
    ]
  }
}
```

### 3d. Usage examples

```
Tell me about transformers #use-local-box
```
→ Forces `local` backend for this session window (5 min TTL).

```
Explain diffusion models in detail #use-ovh
```
→ Routes to OVH Qwen3-32B; falls back to local if OVH is unreachable.

```
Review this code with full reasoning #use-uncle-a
```
→ Routes to real Anthropic API (requires Step 10 + `ANTHROPIC_API_KEY` env var on server).

---

## 4. Optional bearer auth

If `OV_SERVER_API_KEY` is set in the server's environment, all `/v1/messages`
requests require a matching bearer token. Claude Code passes `ANTHROPIC_API_KEY` as
the bearer token automatically, so they must match.

Server side (e.g. `/etc/systemd/system/ov-server.service`):

```ini
[Service]
Environment=OV_SERVER_API_KEY=your-secret-token
```

Client side:

```bash
export ANTHROPIC_API_KEY=your-secret-token
```

---

## 5. LAN access (other machines on the network)

The server listens on `0.0.0.0:11435`. From another machine:

```bash
export ANTHROPIC_BASE_URL=http://EnvyStorm:11435   # or IP
export ANTHROPIC_API_KEY=<token-if-auth-enabled>
claude
```

---

## 6. Observability

Every request is tagged with a 12-hex `X-Request-ID` (Step 15). To correlate logs:

```bash
# See the ID in response headers
curl -si http://localhost:11435/health | grep x-request-id

# Follow a specific request through the journal
sudo journalctl -u ov-server -f | grep abc123def456
```

Log format: `YYYY-MM-DD HH:MM:SS,mmm LEVEL [request_id] message`

Startup lines use `[-]` (no request in flight).

---

## 7. Switching Claude Code back to Anthropic cloud

```bash
unset ANTHROPIC_BASE_URL
# ANTHROPIC_API_KEY must now be your real Anthropic key
claude
```

---

## 8. Backend status

| Backend | Status | Notes |
|---|---|---|
| `local` | Live | qwen3-8b / qwen3-14b on GPU.1 |
| `ovh` | Live | Qwen3-32B via OVH AI Endpoints; needs `OVH_API_KEY` env var |
| `anthropic` | Deferred (Step 10) | Pass-through to api.anthropic.com; needs `ANTHROPIC_API_KEY` env var on server |
