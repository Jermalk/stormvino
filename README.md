# ov_server

OpenVINO-backed OpenAI-compatible API server. Exposes `/v1/chat/completions`, `/v1/embeddings`, and `/v1/models` on port `11435`.

## Starting the server

```bash
python3 /opt/ov_server/ov_server.py
```

With debug request logging:

```bash
python3 /opt/ov_server/ov_server.py --debug
```

## Configuration

Runtime settings live in `config.json` next to `ov_server.py`. The server starts with built-in defaults if the file is absent.

| Key | Default | Description |
|---|---|---|
| `models_dir` | `<script dir>/models` | Directory scanned for OpenVINO LLM models at startup |
| `device` | `GPU.1` | OpenVINO device |
| `default_model` | (first discovered) | Model used when the client does not specify one |
| `agent_model` | (first discovered) | Smaller model used for tool-call / agent turns |
| `embedding_model` | `""` | Embedding model subdirectory name |
| `model_aliases` | `{}` | Map client model names to discovered model names |
| `max_loaded_models` | `2` | Max models kept in VRAM simultaneously |
| `vram_headroom_gb` | `1.5` | Free VRAM required before loading an additional model |
| `max_new_tokens_default` | `2048` | Token cap for normal chat |
| `max_new_tokens_agent` | `200` | Token cap for agent/tool-selection turns |

Models are auto-discovered: any subdirectory of `models_dir` that contains both `openvino_model.xml` and `generation_config.json` is registered as an LLM.

For model conversion, adding new models, and VRAM sizing — see **[MODELS.md](MODELS.md)**.

## Logs

Logs go to the system journal. To follow live:

```bash
journalctl -f _COMM=ov_server
```

## Toggle debug logging without restart

Send `SIGUSR1` to flip debug logging on or off while the server is running:

```bash
kill -USR1 $(pgrep -f ov_server.py)
```

## Network access

The server listens on `0.0.0.0:11435` — accessible from the local network.

| Address | Use |
|---|---|
| `http://EnvyStorm.local:11435` | mDNS hostname (preferred) |
| `http://192.168.0.136:11435` | Direct IP |
| `http://localhost:11435` | Local only |

If the port is blocked, open it with:

```bash
sudo ufw allow 11435/tcp
```

## Health check

```bash
curl -s http://localhost:11435/health | python3 -m json.tool
```

## Available models

```bash
curl -s http://localhost:11435/v1/models | python3 -m json.tool
```

## Example chat request

```bash
curl -s http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-14b-int4-ov", "messages": [{"role": "user", "content": "Hello"}]}' \
  | python3 -m json.tool
```
