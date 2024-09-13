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

The implementation supports four pre-trained models: CodeBERT, GraphCodeBERT, CodeT5, and Unixcoder. Please navigate to the appropriate folder to select the base model you wish to use. Detailed instructions for running each specific pre-trained model with SPENCER can be found in the corresponding folder.

If you want to select CodeBERT as base model, please refer to the [CodeBERT](https://github.com/wcgu1993/SPENCER/tree/main/CodeBERT) folder.

If you want to select GraphCodeBERT as base model, please refer to the [GraphCodeBERT](https://github.com/wcgu1993/SPENCER/tree/main/GraphCodeBERT) folder.

If you want to select CodeT5 as base model, please refer to the [CodeT5](https://github.com/wcgu1993/SPENCER/tree/main/CodeT5) folder.

If you want to select UniXcoder as base model, please refer to the [UniXcoder](https://github.com/wcgu1993/SPENCER/tree/main/UniXcoder) folder.
