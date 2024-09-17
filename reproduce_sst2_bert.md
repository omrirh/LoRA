# SST2 training on BERT using LoRA
#### This repo suggests re-production steps of LoRA performance experiment.
#### It uses GLUE SST2 task for fine-tuning BERT on downstream tasks and testing their results.
#### This experiment is to be scaled to utilize RoBERTa model for training in order to match LoRA SOTA results with SST2
## Pre-requisites:
- Make sure your machine is CUDA-11.3.1 compatible
- Running on Ubuntu >=20.04
- `docker` is installed

## Running the experiment containerized
```bash
make setup-run
```

### After the training & evaluation phases are completed, you should see the session artifacts under `results` under the project path.

