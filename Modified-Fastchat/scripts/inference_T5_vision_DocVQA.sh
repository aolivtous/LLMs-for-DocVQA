CUDA_VISIBLE_DEVICES=0 python /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/fastchat/serve/huggingface_api_inference_vision_DocVQA.py \
    --model_name_or_path google/flan-t5-xl \
    --data_path /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_visualDocVQA_allContext_1.json \
    --output_dir /home/aolivera/TFM-LLM/LLM/Results/inference/val_inference_vision_DocVQA_all_T5w.json \
    --model_max_length 2048 \
    