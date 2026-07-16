SHELL := /bin/bash

.PHONY: demo-generate demo-up demo-seed demo-smoke demo-browser-ready demo-browser-install demo-browser-test demo-down

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

demo-browser-install:
	cd frontend && npm ci && npm exec -- playwright install chromium

demo-browser-test:
	./scripts/demo-browser-test.sh

demo-down:
	./scripts/demo-down.sh
