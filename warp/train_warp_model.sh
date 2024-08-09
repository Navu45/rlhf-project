python warp/train.py \
    --trust_remote_code=True \
    --use_peft=True \
    --seed 42 \
    --lora_r=5 \
    --lora_alpha=16 \
    --model_name=lvwerra/gpt2-imdb \
    --query_dataset=stanfordnlp/imdb \
    --reward_model=sentiment-analysis:lvwerra/distilbert-imdb \
    --ppo_epochs=1 \
    --gradient_checkpointing=False \
    --is_peft_model=True \
    --learning_rate 1.34e-6 \
    --batch_size 64 \
    --mini_batch_size 1 \
    --output_dir=data/warp_gpt2_imdb \
    --steps=100 \
    --iterations=1 \
    # --log_with wandb \

