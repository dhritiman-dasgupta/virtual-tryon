.DEFAULT_GOAL := help
COMFY_ROOT ?= /opt/ComfyUI
PORT ?= 8000

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	 | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install ComfyUI, custom node and Python deps
	COMFY_ROOT=$(COMFY_ROOT) bash scripts/install.sh

models: ## Download and verify the ~19GB of weights
	COMFY_ROOT=$(COMFY_ROOT) bash scripts/download_models.sh

env: ## Create .env from the example
	@test -f .env || (cp .env.example .env && echo "created .env — set API_KEYS before exposing it")

serve: ## Run the API (spawns ComfyUI itself)
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

dev: ## Run with autoreload
	uvicorn app.main:app --host 127.0.0.1 --port $(PORT) --reload

smoke: ## End-to-end check against a running server
	python tests/smoke_test.py

docker-build: ## Build the container
	docker compose build

docker-up: ## Start the container
	docker compose up -d

docker-models: ## Download weights into the container volume
	docker compose run --rm tryon-api bash scripts/download_models.sh

docker-logs: ## Follow logs
	docker compose logs -f

notebook: ## Regenerate the Colab notebook after changing app/
	python3 scripts/make_notebook.py

# --- GPU box -----------------------------------------------------------------
# HOST and PORT change on every new instance; pass them in.
#   make connect HOST=<HOST> PORT=<PORT>

connect: ## Probe the GPU box and report what survived a restart
	HOST=$(HOST) PORT=$(PORT) bash deploy/connect.sh

sync: ## Push app/, pipeline/ and workflows/ to the box and restart the API
	HOST=$(HOST) PORT=$(PORT) RESERVE_VRAM=$(RESERVE_VRAM) bash deploy/sync.sh

box-setup: ## Rebuild the stack on a bare container (~10 min cold)
	ssh -p $(PORT) -i ~/.ssh/id_ed25519 root@$(HOST) \
	  'cd /workspace/swift-teal-stoat && bash setup.sh'

# --- reporting ---------------------------------------------------------------

benchmark: ## Rebuild the benchmark PDF and workbook from the run reports
	python3 reporting/build_benchmark.py --r5 data/runs-5090 --r4 data/runs-4090 --out docs

gallery: ## Rebuild the model+garment=output gallery (offline and S3)
	python3 reporting/build_gallery.py --images data/runs-4090/BEST \
	  --report data/runs-4090/outputs_round4/f6_report.json \
	  --crops data/runs-4090/cache/crops/round4 \
	  --model "data/source-images/female models/model  (6).jpeg" --out docs

test: ## Run the pipeline tests (no GPU needed)
	python3 tests/test_catalogue.py

catalogue: ## Run the full catalogue on the box (HOST/PORT required)
	ssh -p $(PORT) -i ~/.ssh/id_ed25519 root@$(HOST) \
	  'cd /workspace/swift-teal-stoat && ./venv/bin/python pipeline/run_catalogue.py \
	     --model $(or $(MODEL),f6) --guardrail $(or $(GUARDRAIL),default)'

.PHONY: help install models env serve dev smoke notebook test catalogue connect sync box-setup benchmark gallery docker-build docker-up docker-models docker-logs
