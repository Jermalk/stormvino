VENV    = /home/jerzy/ov_env/bin/python
PYTEST  = $(VENV) -m pytest
PTW     = /home/jerzy/ov_env/bin/ptw

test:
	$(PYTEST)

watch:
	$(PTW) -- --tb=short -q
