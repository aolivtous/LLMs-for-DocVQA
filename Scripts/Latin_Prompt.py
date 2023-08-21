import argparse
import json
import math
from torch.utils.data import DataLoader
from SP_DocVQA import singlepage_docvqa_collate_fn
from evaluation_DocVQA import parse_args
from utils import build_dataset, load_config
import tqdm


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
    while len(boxes) > 0:
        line_box = [boxes.pop(0)] #remove box from boxes and add it to line_box
        line_text = [texts.pop(0)] #remove text from texts and add it to line_text
        char_num = len(line_text[-1]) #number of characters of last word in line_text
        line_union_box = line_box[-1] #initialize it to last box in line_box
        while len(boxes) > 0 and line_box[-1][1] == boxes[0][1]: #check if next box is in same line
            line_box.append(boxes.pop(0)) #remove box from boxes and add it to line_box
            line_text.append(texts.pop(0)) #remove text from texts and add it to line_text
            
            line_union_box = union_box(line_union_box, line_box[-1])

            char_dim = (line_box[-1][2] - line_box[-1][0])/len(line_text[-1]) #width of the last character in the line
            if char_dim == 0:
                spaces = 0
            else:
                spaces =  math.ceil((line_box[-1][0] - line_box[-2][2])/char_dim) #number of spaces between last and second last character
            char_num += (len(line_text[-1] )+ spaces) #number of characters of the line

        line_boxes.append(line_box)
        line_texts.append(line_text)
        if char_num >= max_line_char_num:
            max_line_char_num = char_num
            line_width = line_union_box[2] - line_union_box[0]

    char_width = line_width / max_line_char_num
    if char_width == 0:
        char_width = 1

    space_line_texts = []
    for i, line_box in enumerate(line_boxes):
        space_line_text = ""
        for j, box in enumerate(line_box):
            # round always to the next integer
            left_char_num = int(box[0] / char_width)
            #left_char_num = int(box[0] / char_width)
            if left_char_num - len(space_line_text) <= 1:
                space_line_text += " " 
            else:
                #space_line_text += " " * (left_char_num - len(space_line_text))
                space_line_text += "  " + str(left_char_num - len(space_line_text)) + " <spaces> "

            space_line_text += line_texts[i][j]
        space_line_texts.append(space_line_text)

    return space_line_texts

def create_latinLayout(item):
    texts = item["words"]
    text_boxes = item["boxes"]
    sorted_boxes,sorted_texts = boxes_sort(text_boxes,texts)
    space_line_texts = space_layout(texts=sorted_texts, boxes=sorted_boxes)

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
    "prompt_inference": (
        "I have this document with the following words:\n{document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document.\n\n"
        "Answer:"
    ),
    
    "prompt_finetune": (
        "Document:\n{document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document.\n\n"
        "Answer:"
    ),
    }   

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test', help='split to use')
    parser.add_argument('--output_file', type=str, default='test_allData_latin_spaces.json', help='path of the output JSON file that saves the context + questions in the FastChat format and the answers')
    parser.add_argument('--is_inference', type=bool,default=True, help='wheter to include answers or not')
    parser.add_argument('--validQuestions', type=str, default='/home/aolivera/TFM-LLM/LLM/Data/val_validData_BB.json', help='path of the config file')
    args = parser.parse_args()

    config = load_config(parse_args())

    dataset = build_dataset(config, args.split)

    normalizedData = DataLoader(dataset, collate_fn=singlepage_docvqa_collate_fn)
    texts = []
    text_boxes = []
    new_data = []
    alid_questions_id = []
    valid_questions_id = []

    with open(args.validQuestions) as f:
        validData = json.load(f)

    
    for item in validData:
        valid_questions_id.append(int(item['id'].split('_')[1]))
    


    for item in tqdm.tqdm(normalizedData.dataset):

        if len(item['boxes']) != 0: #and item['question_id'] in valid_questions_id:
            space_line_texts = create_latinLayout(item)

            doc = "\n".join(space_line_texts)
            
            #if validData is None:
            context = doc
            query =  context + '. ' + item['questions'] 
        
            if args.is_inference:
                answer = ''
            else :
                #answer = '. '.join(item['answers']) 
                answer = item['answers'][0] 

            conversation = [
                {
                    'from': 'human',
                    'value': query
                },
                {
                    'from': 'gpt',
                    'value': answer
                }
            ]

            new_item = {
                'id': f'identity_{item["question_id"]}',
                'conversations': conversation
            }

            new_data.append(new_item)

    print("total data ", len(normalizedData.dataset))
    print("Valid data ", len(new_data))

    # Convert to JSON string
    new_format_json_str = json.dumps(new_data, indent=2)

    # save the JSON string to a file

    with open(args.output_file, 'w') as f:
        f.write(new_format_json_str)


    

