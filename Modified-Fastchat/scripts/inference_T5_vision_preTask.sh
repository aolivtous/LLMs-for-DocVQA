CUDA_VISIBLE_DEVICES=1 python /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/fastchat/serve/huggingface_api_inference_vision_preTask.py \
    --model_name_or_path lmsys/fastchat-t5-3b-v1.0 \
    --data_path /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_pretaskData_reduced.json \
    --output_dir /home/aolivera/TFM-LLM/LLM/Results/inference/val_inference_vision_Pretask_T5_CLIP_unfrozen_T5g_new_.json \
    --model_max_length 2048 \
    