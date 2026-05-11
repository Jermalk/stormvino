# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 39: Three features shipped. (1) #code/#document/#general chat directives — task_class_directive() in router.py, priority 0 in _detect_signal(). (2) monitor_sidecar.py — independent daemon on :11436, reads VRAM live from /proc/fdinfo + GPU engines + health proxy; VramBar rewritten to use real-time loading gap (liveServerVram − allocatedSum); sidecar systemd unit at /tmp/ov-monitor-sidecar.service needs sudo install if not done. (3) max_new_tokens floor fix — client max_tokens can no longer cap below profile setting; fixes AnythingLLM 200-token cutoff. Next: SVP Phase 4 (Postgres charts).

