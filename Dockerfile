# Start from a base image with Miniconda installed
FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && \
    apt-get install -y sudo libusb-1.0 python3-dev gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /quants-lab

# Create the environment from the environment.yml file (do first to avoid invalidating the environment layer cache)
COPY environment.yml .
# If cchardet fails, we'll install it separately
RUN conda env create -f environment.yml

# Activate the environment and install cchardet separately if it failed
# RUN conda run -n quants_lab pip install cchardet || echo "cchardet installation failed, continuing anyway"

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "quants-lab", "/bin/bash", "-c"]

# Copy and make scripts executable
COPY scripts/ scripts/
RUN chmod +x scripts/*.sh

# Update GLIBCXX if needed (if you get an error about missing GLIBCXX_3.4.32)
# RUN ./scripts/update_glibcxx.sh

# Create wheels directory and handle wheel installation (for utilizing local hummingbot version)
RUN mkdir -p wheels/
COPY wheels/ wheels/

# Optionally install local Hummingbot wheel if present
RUN if [ -n "$(find wheels/ -name 'hummingbot-*.whl' 2>/dev/null)" ]; then \
    echo "Installing local Hummingbot wheel..." && \
    pip install --force-reinstall $(find wheels/ -name 'hummingbot-*.whl') && \
    echo "Local Hummingbot wheel installed successfully"; \
    else \
    echo "No local Hummingbot wheel found, using version from environment.yml"; \
    fi

    

# Uncomment if wanting to deploy somewhere but but since the local volume is mounted in compose file, only useful for remote deployment) 
# COPY config/*.yml config/
# COPY core/ core/
# COPY research_notebooks/ research_notebooks/
# COPY controllers/ controllers/
# COPY tasks/ tasks/
# COPY conf/ conf/

# Default command now uses the task runner
CMD ["conda", "run", "--no-capture-output", "-n", "quants-lab", "python3", "run_tasks.py"]