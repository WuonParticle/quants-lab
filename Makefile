.ONESHELL:
.PHONY: uninstall
.PHONY: install
.PHONY: reference-local-hummingbot

uninstall:
	conda env remove -n quants-lab -y

install:
	conda env create -f environment.yml

# Reference local Hummingbot version use force-repackage=true to force re-package
reference-local-hummingbot:
	bash ./scripts/reference_local_hummingbot.sh $(if $(force-repackage),--force-repackage,)
	
# Build local image
build:
	docker build -t hummingbot/quants-lab -f Dockerfile .
# Run db containers
run-db:
	docker compose -f docker-compose-db.yml up -d

# Stop db containers
stop-db:
	docker compose -f docker-compose-db.yml down

# Define default name
SERVICE_NAME ?= task-runner

# Function to get compose file path
define get_compose_file
$(shell if [ "$(SERVICE_NAME)" != "task-runner" ]; then \
	cp docker-compose-task-runner.yml docker-compose-$(SERVICE_NAME).yml; \
	sed -i 's/^  task-runner:/  $(SERVICE_NAME):/g' docker-compose-$(SERVICE_NAME).yml; \
	echo "docker-compose-$(SERVICE_NAME).yml"; \
else \
	echo "docker-compose-task-runner.yml"; \
fi)
endef

# Run task runner with specified config
run-task:
	@COMPOSE_FILE=$(call get_compose_file); \
	TASK_CONFIG=config/$(config) docker compose -f $$COMPOSE_FILE up $(SERVICE_NAME)
	if [ "$(SERVICE_NAME)" != "task-runner" ]; then rm -f $$COMPOSE_FILE; fi

# Run task runner dettached with specified config
run-task-d:
	@COMPOSE_FILE=$(call get_compose_file); \
	TASK_CONFIG=config/$(config) docker compose -f $$COMPOSE_FILE up -d $(SERVICE_NAME); \
	if [ "$(SERVICE_NAME)" != "task-runner" ]; then rm -f $$COMPOSE_FILE; fi

# Stop task runner
stop-task:
	@COMPOSE_FILE=$(call get_compose_file); \
	docker compose -f $$COMPOSE_FILE down; \
	if [ "$(SERVICE_NAME)" != "task-runner" ]; then rm -f $$COMPOSE_FILE; fi


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
