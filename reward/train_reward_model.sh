# wandb login
# python reward_modeling.py --help
python reward_modeling.py \
    --model_name_or_path=distilbert/distilbert-base-cased \
    --output_dir="reward_modeling_imdb" \
    --per_device_train_batch_size=32 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=1 \
    --eval_strategy="steps" \
    --eval_steps=500 \
    --lora_target_modules "q_lin" "k_lin" "v_lin" "out_lin" "lin1" "lin2" \
    --lora_task_type="SEQ_CLS" \
    --load_in_8bit \
    --load_in_4bit \
    --max_length=512 \