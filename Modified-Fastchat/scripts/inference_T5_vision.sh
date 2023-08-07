CUDA_VISIBLE_DEVICES=0 python /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/fastchat/serve/huggingface_api_inference_vision.py \
    --model_name_or_path google/flan-t5-xl \
    --data_path /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_pretaskData_reduced.json \
    --output_dir /home/aolivera/TFM-LLM/LLM/Results/inference/val_inference_T5_CLIP_unfrozen_T5w_tok.json \
    --model_max_length 2048 \
    