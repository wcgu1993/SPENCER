# SPENCER

This repository contains the code for reproducing the experiments in SPENCER: Self-Adaptive Model Distillation for Efficient Code Retrieval. SPENCER is a framework that employs a recall-rerank pipeline for code retrieval tasks. Specifically, it utilizes dual-encoders for recall and cross-encoders for reranking. Additionally, SPENCER can dynamically adjust the compression ratio of the dual encoders during model distillation based on its impact on the overall performance of the framework. The implementation is based on four pre-trained models: CodeBERT, GraphCodeBERT, CodeT5, and UniXcoder.
## Data Preprocessing

The dataset we adopted is from CodeBERT and originally from CodeSearchNet. You can download and preprocess this dataset using the following command.

```
gdown https://drive.google.com/file/d/1Fd6n_ztivdSErnf382GETHJlaSWRJRVq/view?usp=sharing
unzip data.zip
rm  data.zip
python process_data.py
```

## Appendix

![img](https://github.com/wcgu1993/SPENCER/blob/main/framework.png)
