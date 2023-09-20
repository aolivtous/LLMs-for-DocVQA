CUDA_VISIBLE_DEVICES=5 python /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/fastchat/serve/huggingface_api_inference.py \
    --model /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5_prova_google_clean/checkpoint-369 \
    --data_path /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_validData.json \
    --output_dir /home/aolivera/TFM-LLM/LLM/Results/inference/val_inference_T5_text_fastchat_1epoch.json \
    --model_max_length 2048 \
    