import argparse
import json
import math
from torch.utils.data import DataLoader
from SP_DocVQA import singlepage_docvqa_collate_fn
from evaluation import parse_args
from utils import build_dataset, load_config
import tqdm
#sys.path.append(".")




def boxes_sort_old(boxes): # first sort by y, then sort by x
    """
    Params:
        boxes: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    """
    old_boxes = boxes.copy()
    sorted_id = sorted(range(len(boxes)), key=lambda x: (boxes[x][1], boxes[x][0]))

    # sorted_boxes = [boxes[id] for id in sorted_id]


    return sorted_id


def boxes_sort(boxes,texts): # first sort by y, then sort by x
    """
    Params:
        boxes: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    """

    old_boxes = boxes.copy()
    boxes,ids,texts = sort_by_line(boxes,texts)
    sorted_id = sorted(ids, key=lambda x: (boxes[x][1], boxes[x][0]))

    sorted_boxes = [boxes[id] for id in sorted_id]
    sorted_texts = [texts[id] for id in sorted_id]


    return sorted_boxes, sorted_texts


def sort_by_line(boxes,texts):
    sorted_boxes = []
    sorted_boxes.append(boxes[0])

    for i in range(len(boxes) - 1):
        curr_box = boxes[i]
        next_box = boxes[i+1]
        
        if is_same_line(curr_box, next_box):
            sorted_boxes.append([next_box[0],sorted_boxes[-1][1],next_box[2],sorted_boxes[-1][3]])
        else:
            sorted_boxes.append([next_box[0],next_box[1],next_box[2],next_box[3]])
    
    #sort by y
    sorted_id = sorted(range(len(sorted_boxes)), key=lambda x: boxes[x][1])

    sorted_boxes = [sorted_boxes[id] for id in sorted_id]
    sorted_texts = [texts[id] for id in sorted_id]

    return sorted_boxes,sorted_id,sorted_texts
        


            
        

def union_box(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    return [x1, y1, x2, y2]


def is_same_line(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    
    box1_midy = (box1[1] + box1[3]) / 2
    box2_midy = (box2[1] + box2[3]) / 2

    if box1_midy < box2[3] and box1_midy > box2[1] and box2_midy < box1[3] and box2_midy > box1[1]:
        return True
    else:
        return False

def space_layout(texts, boxes):
    line_boxes = []
    line_texts = []
    max_line_char_num = 0
    line_width = 0
    # print(f"len_boxes: {len(boxes)}")
    while len(boxes) > 0:
        line_box = [boxes.pop(0)] #remove box from boxes and add it to line_box
        line_text = [texts.pop(0)] #remove text from texts and add it to line_text
        char_num = len(line_text[-1]) #number of characters of last word in line_text
        line_union_box = line_box[-1] #initialize it to last box in line_box
        while len(boxes) > 0 and line_box[-1][1] == boxes[0][1]: #check if next box is in same line
            line_box.append(boxes.pop(0)) #remove box from boxes and add it to line_box
            line_text.append(texts.pop(0)) #remove text from texts and add it to line_text
            char_num += 1 + len(line_text[-1]) #number of characters of the line
            line_union_box = union_box(line_union_box, line_box[-1])
        line_boxes.append(line_box)
        line_texts.append(line_text)
        if char_num >= max_line_char_num:
            max_line_char_num = char_num
            line_width = line_union_box[2] - line_union_box[0]
    
    # print(line_width)

    char_width = line_width / max_line_char_num
    print("char width ::::::::::::::::::::::::")
    print(char_width)
    if char_width == 0:
        char_width = 1

    space_line_texts = []
    for i, line_box in enumerate(line_boxes):
        space_line_text = ""
        for j, box in enumerate(line_box):
            # round always to the next integer
            left_char_num = math.ceil(box[0] / char_width)
            #left_char_num = int(box[0] / char_width)
            if left_char_num - len(space_line_text) <= 0:
                space_line_text += " " 
            else:
                space_line_text += " " * (left_char_num - len(space_line_text))

            space_line_text += line_texts[i][j]
        space_line_texts.append(space_line_text)

    return space_line_texts

def create_latinLayout(item,i):
    print(item)
    texts = item["words"]
    text_boxes = item["boxes"]
    sorted_boxes,sorted_texts = boxes_sort(text_boxes,texts)
    #sorted_texts = [texts[id] for id in ids]
  
    #texts = ["{" + f'{count}-{texts[i]}' + "}" for count, i in enumerate(ids)]

    space_line_texts = space_layout(texts=sorted_texts, boxes=sorted_boxes)


    """with open(str(i) + ".txt", "w") as f:
        f.write("\n".join(space_line_texts))"""


    return space_line_texts
    


if __name__ == "__main__":


    PROMPT_DICT = {
    "prompt_task": (
        "You are asked to answer questions asked on a document image.\n"
        "The answers to questions are short text spans taken verbatim from the document. "
        "This means that the answers comprise a set of contiguous text tokens present in the document.\n"
        "Document:\n{document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document with as few words as possible .\n\n"
        "Answer:"
    ),
    "prompt_plain": (
        "Document:\n{document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document.\n\n"
        "Answer:"
    ),
    }   

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val', help='split to use')
    args = parser.parse_args()

    #filepath = "/home/aolivera/SinglePage_DocVQA/train/ocr_results/nmmg0227_4.json"
    #with open(filepath, "r") as f:
    #    data = json.load(f)

     
    config = load_config(parse_args())

    dataset = build_dataset(config, args.split)

    normalizedData = DataLoader(dataset, collate_fn=singlepage_docvqa_collate_fn)
    texts = []
    text_boxes = []
    i = 0
    for item in tqdm.tqdm(normalizedData.dataset):

        space_line_texts = create_latinLayout(item,i)

        doc = "\n".join(space_line_texts)
        text = PROMPT_DICT["prompt_task"].format_map({
            "document": doc,
            "question": "what is the answer?"
        })
        print(text)
        i += 1
        if i == 10:
            break