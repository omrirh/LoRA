FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04


WORKDIR /LoRA


RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git


COPY . .


RUN pip3 install -r requirements.txt


RUN chmod -R 777 /LoRA


CMD ["python3", "run_lora_sst2_roberta_base_experiment.py"]
