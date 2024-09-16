# Base image with CUDA support
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /LoRA

# Install Python, pip, and git
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git

# Copy the repository into the container
COPY . .

# Install Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Ensure proper permissions on the working directory
RUN chmod -R 777 /LoRA

# Run the main script
CMD ["python3", "run_lora_experiment.py"]
