# Docker.md — container management for Open WebUI + SearxNG

## Open WebUI

```
# Create / recreate
docker run -d --name open-webui \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main

# Start / stop / restart
docker start open-webui
docker stop open-webui
docker restart open-webui

# Remove (data is lost — no volume mounted)
docker rm -f open-webui

# Logs
docker logs open-webui --tail 50 --follow

# Update image
docker pull ghcr.io/open-webui/open-webui:main
docker rm -f open-webui
# then recreate with docker run above
```

**URL:** http://localhost:3000  
**ov_server endpoint configured inside:** http://host.docker.internal:11435

---

## SearxNG

```
# Create / recreate
docker run -d --name searxng \
  -p 8080:8080 \
  -v ~/searxng-settings.yml:/etc/searxng/settings.yml \
  searxng/searxng

# Start / stop / restart
docker start searxng
docker stop searxng
docker restart searxng

# Remove
docker rm -f searxng

# Logs
docker logs searxng --tail 50 --follow

# Test JSON search (must return results)
curl -s "http://localhost:8080/search?q=test&format=json" | python3 -m json.tool | head -10
```

**URL:** http://localhost:8080  
**Open WebUI setting:** Web Search Engine → SearxNG, URL → `http://172.17.0.1:8080/search?q=<query>&format=json`  
**Settings file:** `~/searxng-settings.yml` — must have `json` under `search.formats` or SearxNG returns 403

---

## General commands

```
# List running containers
docker ps

# List all containers including stopped
docker ps -a

# List images
docker images

# Remove stopped containers
docker container prune

# Remove unused images
docker image prune
```
