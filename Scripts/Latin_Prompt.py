import os
import json
import sys
sys.path.append(".")



def boxes_sort(boxes):
    """
    Params:
        boxes: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    """
    sorted_id = sorted(range(len(boxes)), key=lambda x: (boxes[x][1], boxes[x][0]))

    # sorted_boxes = [boxes[id] for id in sorted_id]


    return sorted_id


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
        line_box = [boxes.pop(0)]
        line_text = [texts.pop(0)]
        char_num = len(line_text[-1])
        line_union_box = line_box[-1]
        while len(boxes) > 0 and is_same_line(line_box[-1], boxes[0]):
            line_box.append(boxes.pop(0))
            line_text.append(texts.pop(0))
            char_num += len(line_text[-1])
            line_union_box = union_box(line_union_box, line_box[-1])
        line_boxes.append(line_box)
        line_texts.append(line_text)
        if char_num >= max_line_char_num:
            max_line_char_num = char_num
            line_width = line_union_box[2] - line_union_box[0]
    
    # print(line_width)

    char_width = line_width / max_line_char_num
    print(char_width)
    if char_width == 0:
        char_width = 1

    space_line_texts = []
    for i, line_box in enumerate(line_boxes):
        space_line_text = ""
        for j, box in enumerate(line_box):
            left_char_num = int(box[0] / char_width)
            space_line_text += " " * (left_char_num - len(space_line_text))
            space_line_text += line_texts[i][j]
        space_line_texts.append(space_line_text)

    return space_line_texts


if __name__ == "__main__":

    filepath = "/home/xxx/workspace/VrDU/datas/funsd/testing_data/annotations/82092117.json"
    with open(filepath, "r") as f:
        data = json.load(f)
    texts = []
    text_boxes = []

    for i, item in enumerate(data["form"]):
        texts.append(item["text"])
        # texts.append("{" + f'{i}-{item["text"]}' + "}")
        text_boxes.append(item["box"])
    ids = boxes_sort(text_boxes)
    texts = ["{" + f'{count}-{texts[i]}' + "}" for count, i in enumerate(ids)]
    text_boxes = [text_boxes[i] for i in ids]
    space_line_texts = space_layout(texts=texts, boxes=text_boxes)
    with open("82092117.txt", "w") as f:
        f.write("\n".join(space_line_texts))