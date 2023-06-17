
import argparse

from fastchat.utils import clean_flant5_ckpt





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dirWeights", type=str, default="/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/scripts/checkpoints/checkpoints_flant5") 
   
    args = parser.parse_args()

    clean_flant5_ckpt(args.dirWeights)
  




