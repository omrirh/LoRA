FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

WORKDIR /LoRA

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git

COPY requirements.txt .
COPY run_lora_sst2_bert_base_experiment.py .

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir scikit-learn==1.3.2



RUN chmod -R 777 /LoRA

CMD ["python3", "run_lora_sst2_bert_base_experiment.py"]
