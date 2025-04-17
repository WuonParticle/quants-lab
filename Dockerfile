# Start from a base image with Miniconda installed
FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && \
    apt-get install -y sudo libusb-1.0 python3-dev gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /quants-lab

# Copy the current directory contents and the Conda environment file into the container
COPY core/ core/
COPY environment.yml .
COPY research_notebooks/ research_notebooks/
COPY controllers/ controllers/
COPY tasks/ tasks/
COPY conf/ conf/

# Create the environment from the environment.yml file
# If cchardet fails, we'll install it separately
RUN conda env create -f environment.yml

# Activate the environment and install cchardet separately if it failed
# RUN conda run -n quants_lab pip install cchardet || echo "cchardet installation failed, continuing anyway"

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "quants-lab", "/bin/bash", "-c"]

# Copy task configurations
COPY config/tasks.yml /quants-lab/config/tasks.yml

# Copy and make scripts executable
COPY scripts/ scripts/
RUN chmod +x scripts/*.sh

# Update GLIBCXX if needed (if you get an error about missing GLIBCXX_3.4.32)
# RUN ./scripts/update_glibcxx.sh

# Create wheels directory and handle wheel installation
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

# Default command now uses the task runner
CMD ["conda", "run", "--no-capture-output", "-n", "quants-lab", "python3", "run_tasks.py"]