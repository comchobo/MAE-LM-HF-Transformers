# MAE-LM-HF-Transformers

This is **unofficial HF-transformers implementation** of [MAE-LM](https://github.com/yumeng5/MAE-LM). You can read the paper in [Arxiv](https://arxiv.org/abs/2302.02060)

Citation block is as follows.
```
@inproceedings{meng2024maelm,
  title={Representation Deficiency in Masked Language Modeling},
  author={Meng, Yu and Krishnan, Jitin and Wang, Sinong and Wang, Qifan and Mao, Yuning and Fang, Han and Ghazvininejad, Marjan and Han, Jiawei and Zettlemoyer, Luke},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
--------------------

This repository contains the scripts for pretraining MAE-LM, Representation Deficiency in Masked Language Modeling, which is compatiable with Huggingface Transformers library. You guys can easily mimic pre-training method in MAE-LM, peeking `train.py`

Since pre-trained model class is inherited from RoBERTa, pre-trained models are able to be loaded the very simple code we all know, as follows:

```python

from transformers import RobertaPreTrainedModel
model = RobertaPreTrainedModel.from_pretrained("your_pre-trained_model")

```
