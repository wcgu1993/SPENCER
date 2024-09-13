# SPENCER

This repository contains the code for reproducing the experiments in SPENCER: Self-Adaptive Model Distillation for Efficient Code Retrieval. SPENCER is a framework that employs a recall-rerank pipeline for code retrieval tasks. Specifically, it utilizes dual-encoders for recall and cross-encoders for reranking. Additionally, SPENCER can dynamically adjust the compression ratio of the dual encoders during model distillation based on its impact on the overall performance of the framework. The implementation is based on four pre-trained models: CodeBERT, GraphCodeBERT, CodeT5, and UniXcoder.

![img](https://github.com/wcgu1993/SPENCER/blob/main/framework.png)

## Dependency

- pip install torch
- pip install transformers

## Data Preprocessing

The dataset we adopted is from CodeBERT and originally from CodeSearchNet. You can download and preprocess this dataset using the following command.

```
gdown https://drive.google.com/file/d/1Fd6n_ztivdSErnf382GETHJlaSWRJRVq/view?usp=sharing
unzip data.zip
rm  data.zip
python process_data.py
```

## Run SPENCER

The implementation supports four pre-trained models: CodeBERT, GraphCodeBERT, CodeT5, and UniXcoder. To select the base model, navigate to the corresponding folder for detailed instructions on running the model with SPENCER.
-For CodeBERT,, please refer to the [CodeBERT](https://github.com/wcgu1993/SPENCER/tree/main/CodeBERT) folder.
-For GraphCodeBERT, please refer to the [GraphCodeBERT](https://github.com/wcgu1993/SPENCER/tree/main/GraphCodeBERT) folder.
-For CodeT5, please refer to the [CodeT5](https://github.com/wcgu1993/SPENCER/tree/main/CodeT5) folder.
-For UniXcoder, please refer to the [UniXcoder](https://github.com/wcgu1993/SPENCER/tree/main/UniXcoder) folder.
