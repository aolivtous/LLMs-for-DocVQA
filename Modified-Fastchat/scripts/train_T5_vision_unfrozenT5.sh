CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port=9774 /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/fastchat/train/train_flant5_vision_unfrozenT5.py \
    --model_name_or_path lmsys/fastchat-t5-3b-v1.0\
    --data_path /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/train_visualDocVQA_allContext_CL.json\
    --bf16 True \
    --output_dir /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5_T5DecUnfrozen_allWords_fastchat\
    --num_train_epochs 30 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --metric_for_best_model "eval_loss" \
    --preprocessed_path /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/preprocessed_data/train_T5unfrozen_allWords_CL.json \
    --gradient_checkpointing True \