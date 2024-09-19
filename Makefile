IMAGE_NAME = lora-bert-base-sst2-experiment
DOCKERFILE = Dockerfile
HOST_RESULTS_DIR = /mnt/results

build:
	@echo "Building Docker image: $(IMAGE_NAME)..."
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

setup-gpu:
	@echo "Setting up GPU drivers and NVIDIA Docker..."
	sudo apt-get update
	sudo apt-get install -y nvidia-docker2
	sudo systemctl restart docker
	@echo "Verifying GPU setup..."
	docker run --rm --gpus all nvidia/cuda:11.3.1-base nvidia-smi

experiment:
	@echo "Running the Docker container with GPU utilization..."
	docker run --gpus all -it -v $(HOST_RESULTS_DIR):/LoRA/results $(IMAGE_NAME)

setup-run:
	make setup-gpu
	make build
	make experiment
