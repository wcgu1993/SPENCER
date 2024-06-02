# SPENCER

This repository cotains the code of SPENCER, which is a code retrieval framework with the combination of cross encoder and distilled dual encoder. The impletation of this framework based on four pre-trained models including CodeBERT, GraphCodeBERT, CodeT5, and UniXcoder. 

## Data Preprocessing

The dataset we adopted is from CodeBERT and originally from CodeSearchNet. You can download and preprocess this dataset using the following command.

```
gdown https://drive.google.com/file/d/1Fd6n_ztivdSErnf382GETHJlaSWRJRVq/view?usp=sharing
unzip data.zip
rm  data.zip
python process_data.py
```

## Appendix

![img](https://github.com/wcgu1993/SPENCER/blob/main/Table5.png)
