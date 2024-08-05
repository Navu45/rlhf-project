# wandb login
# python reward_modeling.py --help
python reward/train.py \
    --model_name_or_path=distilbert/distilbert-base-cased \
    --output_dir="reward_modeling_imdb_peft" \
    --per_device_train_batch_size=16 \
    --max_steps=10000 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=True \
    --learning_rate=6.41e-6 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=1 \
    --eval_strategy="steps" \
    --eval_steps=1000 \
    --max_length=512 \
    --lr_scheduler_type="cosine_with_min_lr" \
    --lr_scheduler_kwargs='{"min_lr": 1e-9}' \
    --save_strategy=steps \
    --save_steps=1000 \
    --save_total_limit=5 \
    --load_best_model_at_end=True \
    --fp16=True \
    --max_steps=3 \
    --use_peft=True \
    --lora_target_modules "q_lin" "k_lin" "v_lin" "out_lin" \
    --lora_task_type="SEQ_CLS" \
    --auto_find_batch_size=True \
    --lora_modules_to_save "classifier.bias" "classifier.weight" "pre_classifier.bias" "pre_classifier.weight" \
