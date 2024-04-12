HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIFFUSERS_OFFLINE=1

torchrun --nproc_per_node 1 --standalone train_dreambooth_colossalai.py \
  --pretrained_model_name_or_path="./model/stable-diffusion-v1-4"  \
  --instance_data_dir="./conan" \
  --output_dir="./weight_output" \
  --instance_prompt="a photo of kudoshinichi" \
  --resolution=512 \
  --plugin="gemini" \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000
