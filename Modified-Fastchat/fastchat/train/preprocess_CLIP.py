import json
import torch
from transformers import CLIPImageProcessor 
from PIL import Image
import tqdm

from transformers import CLIPVisionModel


if __name__ == "__main__":

    output_dir = "/data/users/aolivera/preprocessCLIP/train/"
    data_path = "/home/aolivera/TFM-LLM/LLM/Modified-Fastchat/playground/data/train_pretaskData_reduced.json"


    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    
    with open(data_path) as f:
        data = json.load(f)
    

    output_data = []

    for d in tqdm.tqdm(data):
        image = Image.open(d['image']).resize((224, 224))
        img = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        img_embeds = vision_tower(img.unsqueeze(0), output_hidden_states=True)
        img_embeds = img_embeds.hidden_states[-1][:, :1,:]
        
        #save torch tensor
        torch.save(img_embeds, output_dir + d['image'].split('/')[-1].split('.')[0] + '.pt')

        #save 
        print("")
        
      





