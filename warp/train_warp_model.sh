python warp/train.py \
    --trust_remote_code=True \
    --use_peft=True \
    --seed 42 \
    --lora_r=5 \
    --lora_alpha=16 \
    --model_name=lvwerra/gpt2-imdb \
    --query_dataset=stanfordnlp/imdb \
    --reward_model=text-classification:lvwerra/distilbert-imdb \
    --ppo_epochs=1 \
    --gradient_checkpointing=True \
    --is_peft_model=True \
    --learning_rate 1.41e-5 \
    --batch_size 1 \
    --mini_batch_size 1 \
    # --log_with wandb \

