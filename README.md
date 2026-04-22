# ov_server

OpenVINO-backed OpenAI-compatible API server. Exposes `/v1/chat/completions`, `/v1/embeddings`, and `/v1/models` on port `11435`.

## Starting the server

```bash
python3 /ov_server/ov_server.py
```

With debug request logging:

```bash
python3 /ov_server/ov_server.py --debug
```

## Logs

Logs go to the system journal. To follow live:

```bash
journalctl -f _COMM=python3 | grep -E "DEBUG|INFO|ERROR"
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
  -d '{"model": "qwen3-14b-int4", "messages": [{"role": "user", "content": "Hello"}]}' \
  | python3 -m json.tool
```
