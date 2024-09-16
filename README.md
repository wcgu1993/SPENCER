# SPENCER

This repository contains the code for reproducing the experiments in SPENCER: Self-Adaptive Model Distillation for Efficient Code Retrieval. SPENCER is a framework that employs a recall-rerank pipeline for code retrieval tasks. Specifically, it utilizes dual-encoders for recall and cross-encoders for reranking. Additionally, SPENCER can dynamically adjust the compression ratio of the dual encoders during model distillation based on its impact on the overall performance of the framework. The implementation is based on four pre-trained models: CodeBERT, GraphCodeBERT, CodeT5, and UniXcoder.

![img](https://github.com/wcgu1993/SPENCER/blob/main/framework.png)

Above figure is the overall framework of SPENCER. Initially, both the dual-encoder and cross-encoder are trained. After training, code snippets in the code database are encoded into vectors using the code-encoder, which is a dual-encoder. These code vectors are then stored in the database. When a user query is received, it is processed by the query encoder, another dual-encoder, to generate a query representation vector. To identify the most relevant code candidates, cosine similarity is computed between the query vector and each code vector in the database, and the results are sorted in descending order. The top K code candidates with the highest cosine similarity are retrieved. Each candidate is then combined with the original query to form a new concatenated input, which is fed into the cross-encoder. The cross-encoder re-ranks the code candidates based on matching scores, sorted in descending order. Finally, the top K re-ranked code candidates are combined with the remaining code candidates from the dual-encoder, and this combined list is returned as the final code list to the user.

![img](https://github.com/wcgu1993/SPENCER/blob/main/model_distillation.png)

The figure above illustrates the model distillation process in SPENCER. SPENCER utilizes two main components for this process: Distillation with a Teaching Assistant and Query Encoder Distillation.

**Distillation with Teaching Assistant**: To address the learning challenges arising from the size disparity between the large teacher model and the small student model, we first train an intermediate teaching assistant model. Both the teacher model and the teaching assistant model are then used to train the student model. Details on the selection strategy for the teaching assistant model will be provided in the original paper.

**Query Encoder Distillation**: This process involves distilling a smaller query encoder from the original query encoder using both single modality and dual modality training losses. The single modality loss aligns the output of the distilled query encoder with that of the original query encoder. The dual modality loss, on the other hand, ensures that the distilled query encoder learns the relative positional relationships between the outputs of the original query encoder and the code encoder.

## Dependency

- pip install torch
- pip install transformers

## Data Preprocessing

The dataset we adopted is from CodeBERT and originally from CodeSearchNet. You can download and preprocess this dataset using the following command.

```
gdown https://drive.google.com/file/uc?id=1FuZmzJaVzyv5D7HAXka8RloSTqv7WXtj
unzip data.zip
rm  data.zip
python process_data.py
```

## Run SPENCER

The implementation supports four pre-trained models: CodeBERT, GraphCodeBERT, CodeT5, and UniXcoder. To select the base model, navigate to the corresponding folder for detailed instructions on running the model with SPENCER.

- For CodeBERT, please refer to the [CodeBERT](https://github.com/wcgu1993/SPENCER/tree/main/CodeBERT) folder.
- For GraphCodeBERT, please refer to the [GraphCodeBERT](https://github.com/wcgu1993/SPENCER/tree/main/GraphCodeBERT) folder.
- For CodeT5, please refer to the [CodeT5](https://github.com/wcgu1993/SPENCER/tree/main/CodeT5) folder.
- For UniXcoder, please refer to the [UniXcoder](https://github.com/wcgu1993/SPENCER/tree/main/UniXcoder) folder.
