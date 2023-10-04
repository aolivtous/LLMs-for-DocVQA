![Alt text](aolivtous/LLM/Figures/VisualMethod.png)

# Large Language Models for Document Visual Question Answering

This study introduces innovative methods for Document Visual Question Answering (DocVQA) through the utilization of Large Language Models (LLMs). Our approach involves fine-tuning the Flan-T5 model on the SP-DocVQA dataset with diverse context types, revealing the effectiveness of incorporating spatial information. By utilizing both the document’s textual content and the corresponding bounding box locations of words, we achieve the best performance, reaching an ANLS score of 0.76, using only the text modality. Furthermore, we attempt to incorporate word recognition in the language model itself. To this purpose, we
present a multimodal DocVQA pipeline and establish a pre-training task aimed at aligning the visual features of cropped word images with the LLM space. This approach enables the LLM to effectively understand and process visual information. Finally, we explore two novel methods for performing DocVQA by utilizing the visual embeddings of words. These approaches represent initial steps toward developing a comprehensive and robust solution for addressing this challenging task in an end-to-end manner.

