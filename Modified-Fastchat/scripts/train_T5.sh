CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --use-env --nproc_per_node=2 --master_port=9778 /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/fastchat/train/train_flant5.py \
    --model_name_or_path google/flan-t5-xl \
    --data_path /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/train_validData_BB.json \
    --bf16 True \
    --output_dir /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/checkpoints/checkpoints_T5_textBB_05 \
    --num_train_epochs 0.4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap T5Block \
    --tf32 True \
    --model_max_length 2048 \
    --preprocessed_path ./preprocessed_data/processed_text_BB.json \
    --gradient_checkpointing True 
