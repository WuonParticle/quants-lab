.ONESHELL:
.PHONY: uninstall
.PHONY: install
.PHONY: reference-local-hummingbot

# Suppress Docker Compose orphan warnings
export COMPOSE_IGNORE_ORPHANS=true

# Define default name
service_name ?= task-runner

# Default VPN files
vpn_config ?= ~/vpn/config.ovpn
vpn_auth ?= ~/vpn/auth.txt
timescale_host ?= timescaledb
mongo_host ?= mongodb
docker_network ?= hummingbot-vpn

uninstall:
	conda env remove -n quants-lab -y

install:
	conda env create -f environment.yml

# See reference_local_hummingbot.sh for available options
reference-local-hummingbot:
	bash ./scripts/reference_local_hummingbot.sh $(if $(force-repackage),--force-repackage,) $(ARGS)
	
# Build local image
build:
	docker build -t hummingbot/quants-lab -f Dockerfile .
	
# Run db containers
run-db:
	docker network inspect $(docker_network) >/dev/null 2>&1 || docker network create $(docker_network)
	docker compose -f docker-compose-db.yml up -d

# Stop db containers
stop-db:
	docker compose -f docker-compose-db.yml down

# Function to get compose file path
define get_compose_file
$(shell \
if [ "$(service_name)" != "task-runner" ]; then \
	cp docker-compose-task-runner.yml docker-compose-$(service_name).yml; \
	sed -i 's/^  task-runner:/  $(service_name):/g' docker-compose-$(service_name).yml; \
	sed -i 's/^  vpn-task-runner:/  vpn-$(service_name):/g' docker-compose-$(service_name).yml; \
	echo "docker-compose-$(service_name).yml"; \
else \
	echo "docker-compose-task-runner.yml"; \
fi)
endef

# Define which service to use (regular or VPN version)
define get_service_name
$(shell \
if [ "$(use_vpn)" = "true" ]; then \
	echo "vpn-$(service_name)"; \
else \
	echo "$(service_name)"; \
fi)
endef

# Run task runner with specified config
run-task:
	@COMPOSE_FILE=$(call get_compose_file); \
	SERVICE_TO_USE=$(call get_service_name); \
	TASK_CONFIG=config/$(config) docker compose -f $$COMPOSE_FILE up $$SERVICE_TO_USE; \
	if [ "$(service_name)" != "task-runner" ]; then rm -f $$COMPOSE_FILE; fi

# Run task runner detached with specified config
run-task-d:
	@COMPOSE_FILE=$(call get_compose_file); \
	SERVICE_TO_USE=$(call get_service_name); \
	TASK_CONFIG=config/$(config) docker compose -f $$COMPOSE_FILE up -d $$SERVICE_TO_USE; \
	if [ "$(service_name)" != "task-runner" ]; then rm -f $$COMPOSE_FILE; fi

# Stop task runner
stop-task:
	@COMPOSE_FILE=$(call get_compose_file); \
	SERVICE_TO_USE=$(call get_service_name); \
	docker compose -f $$COMPOSE_FILE down; \
	if [ "$(service_name)" != "task-runner" ]; then rm -f $$COMPOSE_FILE; fi

# Start a VPN container
start-vpn:
	@config_file=$$(echo "$(vpn_config)" | sed 's|^~|$(HOME)|'); \
	auth_file=$$(echo "$(vpn_auth)" | sed 's|^~|$(HOME)|'); \
	[ -f "$$config_file" ] || { echo "Error: VPN config not found: $$config_file"; exit 1; }; \
	[ -f "$$auth_file" ] || { echo "Error: VPN auth not found: $$auth_file"; exit 1; }; \
	docker network inspect $(docker_network) >/dev/null 2>&1 || docker network create $(docker_network); \
	docker build -t quants-lab-vpn -f Dockerfile.vpn .; \
	docker rm -f vpn-container 2>/dev/null || true; \
	VPN_CONFIG_PATH=$$config_file \
	VPN_AUTH_PATH=$$auth_file \
	TIMESCALE_HOST=$(timescale_host) \
	MONGO_HOST=$(mongo_host) \
	DOCKER_NETWORK=$(docker_network) \
	docker compose -f docker-compose-vpn.yml up -d

# Stop the VPN container
stop-vpn:
	@docker compose -p quants-lab -f docker-compose-vpn.yml down 2>/dev/null || true

# Run multiple instances of task runner in parallel
run-parallel:
	@if [ -z "$(instances)" ]; then \
		echo "Usage: make run-parallel instances=<number_of_instances> config=<task_config_file>"; \
		exit 1; \
	fi
	@mkdir -p tmp
	@echo 'services:' > tmp/docker-compose.worker-override.yml
	@for i in $$(seq 0 $$(expr $(instances) - 1)); do \
		echo "  worker-$$i:" >> tmp/docker-compose.worker-override.yml; \
		echo "    extends:" >> tmp/docker-compose.worker-override.yml; \
		echo "      service: worker" >> tmp/docker-compose.worker-override.yml; \
		echo "      file: ../docker-compose-parallel.yml" >> tmp/docker-compose.worker-override.yml; \
		echo "    container_name: qlp-worker-$$i" >> tmp/docker-compose.worker-override.yml; \
		echo "    environment:" >> tmp/docker-compose.worker-override.yml; \
		echo "      - WORKER_ID=$$i" >> tmp/docker-compose.worker-override.yml; \
	done
	TASK_CONFIG=config/$(config) TOTAL_WORKERS=$(instances) docker compose -p quants-lab-parallel -f tmp/docker-compose.worker-override.yml up -d

# Stop parallel task runners
stop-parallel:
	docker compose -p quants-lab-parallel -f tmp/docker-compose.worker-override.yml down
	@rm -f tmp/docker-compose.worker-override.yml
