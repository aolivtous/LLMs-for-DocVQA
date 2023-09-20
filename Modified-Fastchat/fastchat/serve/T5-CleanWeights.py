
import argparse

from fastchat.utils import clean_flant5_ckpt
import os





if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirWeights", type=str, default="/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5_prova_google/checkpoint-738") 
   
    args = parser.parse_args()

    clean_flant5_ckpt(args.dirWeights)
  




