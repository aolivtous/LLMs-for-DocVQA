import argparse
import json
import tqdm
from PIL import Image
import os

def get_box_crop(box):
    # Extract x and y values from the coordinates

    x = [box[0], box[2], box[4], box[6]]
    y = [box[1], box[3], box[5], box[7]]

    # Get the min and max x and y values
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

    if x_min ==  x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1


    return [x_min, y_min, x_max, y_max]

        

def get_dataset_visual(data_dir, output_dir):
    
    doc_dict = {}

    for docImg in tqdm.tqdm(os.listdir(os.path.join(data_dir, "documents"))):

        gt_imgs = []
        gt_words = []


        #get the image in black and white
        im = Image.open(os.path.join(data_dir, "documents", docImg)).convert('L')

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        doc_name = docImg.split('.')[0]
        document_data = json.load(open(os.path.join(data_dir, "ocr_results", doc_name + '.json')))

        #create the directory
        if not os.path.exists(os.path.join(output_dir, doc_name)):
            os.makedirs(os.path.join(output_dir, doc_name))

        i = 0
        for line in tqdm.tqdm(document_data['recognitionResults'][0]['lines']):
            
            for word in line['words']:
                gt_words.append(word['text'])

                #get the bounding box
                box = word['boundingBox']
                #copy the image
                img = im.copy()

                #crop the image
                box_crop = get_box_crop(box)
                
                    
                cropped_img = img.crop((box_crop[0], box_crop[1], box_crop[2], box_crop[3]))

                img_dir = os.path.join(output_dir, doc_name, doc_name + '_' + str(i) + '.png')
                gt_imgs.append(img_dir)
                cropped_img.save(img_dir)

                i += 1

        doc_dict[doc_name] = {"words": gt_words , "images": gt_imgs}

        # save gt to txt
        with open(os.path.join(output_dir, doc_name, doc_name + '.txt'), 'w') as f:
            for item in gt_words:
                f.write("%s\n" % item)
    
    # save gt to json
    with open(os.path.join(output_dir, 'gt.json'), 'w') as f:
        json.dump(doc_dict, f)


    

if __name__ == "__main__":

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val', help='split to use')
    parser.add_argument('--data_dir', type=str, default='/home/aolivera/SinglePage_DocVQA', help='path of the dataset')
    parser.add_argument('--output_dir', type=str, default='/home/aolivera/SinglePage_crops', help='path of to save the dataset')
    args = parser.parse_args()


    # Convert to new format
    get_dataset_visual(os.path.join(args.data_dir, args.split), os.path.join(args.output_dir, args.split))

   

