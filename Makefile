SHELL := /bin/bash

.PHONY: demo-generate demo-up demo-seed demo-smoke demo-browser-ready demo-down

demo-generate:
	./scripts/demo-generate.sh

demo-up:
	./scripts/demo-up.sh

demo-seed:
	./scripts/demo-seed.sh

demo-smoke:
	./scripts/demo-smoke.sh

demo-browser-ready:
	./scripts/demo-browser-ready.sh

demo-down:
	./scripts/demo-down.sh
