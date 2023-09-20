CUDA_VISIBLE_DEVICES=5 python /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/fastchat/serve/huggingface_api_inference_vision_preTask.py \
    --model_name_or_path google/flan-t5-xl \
    --data_path /home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/val_pretaskData_reduced_test_1000.json \
    --output_dir /home/aolivera/TFM-LLM/LLM/Results/inference/val_TEST_inference_vision_Pretask_T5_CLIP_unfrozen_T5_flant_w_New.json \
    --model_max_length 2048 \
    