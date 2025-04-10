.ONESHELL:
.PHONY: uninstall
.PHONY: install


uninstall:
	conda env remove -n quants-lab -y

install:
	conda env create -f environment.yml
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
service_name ?= task-runner

# Function to get compose file path
define get_compose_file
$(shell \
if [ "$(service_name)" != "task-runner" ]; then \
	cp docker-compose-task-runner.yml docker-compose-$(service_name).yml; \
	sed -i 's/^  task-runner:/  $(service_name):/g' docker-compose-$(service_name).yml; \
	echo "docker-compose-$(service_name).yml"; \
else \
	echo "docker-compose-task-runner.yml"; \
fi)
endef

# Run task runner with specified config
run-task:
	@COMPOSE_FILE=$(call get_compose_file); \
	TASK_CONFIG=config/$(config) docker compose -f $$COMPOSE_FILE up $(service_name)
	if [ "$(service_name)" != "task-runner" ]; then rm -f $$COMPOSE_FILE; fi

# Run task runner dettached with specified config
run-task-d:
	@COMPOSE_FILE=$(call get_compose_file); \
	TASK_CONFIG=config/$(config) docker compose -f $$COMPOSE_FILE up -d $(service_name); \
	if [ "$(service_name)" != "task-runner" ]; then rm -f $$COMPOSE_FILE; fi

# Stop task runner
stop-task:
	@COMPOSE_FILE=$(call get_compose_file); \
	docker compose -f $$COMPOSE_FILE down; \
	if [ "$(service_name)" != "task-runner" ]; then rm -f $$COMPOSE_FILE; fi

# Add this target after the run-task target
run-single-task:
	python run_single_task.py --config config/$(config)
