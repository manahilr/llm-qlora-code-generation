install:
	pip install -r requirements.txt

data:
	python data/prepare_data.py

train:
	python train.py --config configs/qlora_config.yaml

baseline:
	python baseline.py --config configs/qlora_config.yaml

eval:
	python evaluate.py --config configs/qlora_config.yaml

push:
	huggingface-cli upload YOUR-USERNAME/llama-3-1-8b-code-qlora ./outputs

.PHONY: install data train baseline eval push