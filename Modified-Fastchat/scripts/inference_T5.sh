CUDA_VISIBLE_DEVICES=0 python /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/fastchat/serve/huggingface_api_inference.py \
    --model_name_or_path /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5_8epochs \
    --data_path /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_validData.json \
    --output_dir /home/aolivera/TFM-LLM/LLM/Results/inference/val_inference_T5_text_newVersionCheck.json \
    --model_max_length 2048 \
    