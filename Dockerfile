# Base image with CUDA support
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /LoRA

# Copy the repository into the container
COPY . .

# Install Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Ensure proper permissions on the working directory
RUN chmod -R 777 /LoRA

# Run the main script
CMD ["/bin/bash", "-c", "python", "run_lora_experiment.py"]
