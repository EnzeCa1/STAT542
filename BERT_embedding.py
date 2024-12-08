import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from transformers import BertTokenizer, BertModel
from IPython.core.display import HTML
import torch
import os
import time
import warnings

def get_bert_embeddings(reviews, model, tokenizer):
    model.to('cpu')
    model.eval()
    embeddings = []
    with torch.no_grad():
        for review in reviews:
            tokens = tokenizer(review, return_tensors='pt', truncation=True, padding=True, max_length=512).to('cpu')
            outputs = model(**tokens)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding[0])
    return np.array(embeddings)

#prepare and align bert embeddings for interpretability 
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

sample_reviews = train_data['review'].sample(500, random_state=999).values
bert_embeddings = get_bert_embeddings(sample_reviews, bert_model, tokenizer)
openai_embeddings = train_data.iloc[:, 3:].sample(500, random_state=999).values

alignment_model = LinearRegression()
alignment_model.fit(bert_embeddings, openai_embeddings) 