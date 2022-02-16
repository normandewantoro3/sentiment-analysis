import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from model.py import DocumentSentimentDataset

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv

class BertSentiment:
    def __init__(self):
        # Load Tokenizer and Config
        self.tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        config = BertConfig.from_pretrained('indobenchmark/indobert-base-p1')
        config.num_labels = DocumentSentimentDataset.NUM_LABELS

        self.model = BertForSequenceClassification(config=config)
        self.model.load_state_dict(torch.load('bert-model/indonlu_pretrained.pth', map_location=torch.device('cpu')))
        self.model.eval()
        
    def BERT_sentiment(self, text, type = "raw"):
        subwords = self.tokenizer.encode(text)
        subwords = torch.LongTensor(subwords).view(1, -1).to(self.model.device)
        logits = self.model(subwords)[0]
        label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
        l = F.softmax(logits, dim=-1).squeeze().tolist()
        return [self.label_sentiment((l[0] - l[2])/sum(l))]
    
    def label_sentiment(self, score, cutoff = [-0.1, 0.1]):
            if score > cutoff[0]:
                return "pos"
            elif score < cutoff[1]:
                return "neg"
            else:
                return "neu"
    
    
class VaderSentiment:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        # # Loughran and McDonald
        positive = []
        with open('lm_positive.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                positive.append(row[0].strip())

        negative = []
        with open('lm_negative.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                entry = row[0].strip().split(" ")
                if len(entry) > 1:
                    negative.extend(entry)
                else:
                    negative.append(entry[0])

        final_lex = {}
        final_lex.update({word:3.0 for word in positive})
        final_lex.update({word:-3.0 for word in negative})
        final_lex.update(self.sia.lexicon)
        self.sia.lexicon = final_lex
    
    def VADER_sentiment(self, text, type = "raw"): 
        res = self.sia.polarity_scores(text)
        if type == "raw":
            return res['compound']
        if type == "label":
            return [self.label_sentiment(res['compound'])]
            
    def label_sentiment(self, score, cutoff = [-0.1, 0.1]):
            if score > cutoff[0]:
                return "pos"
            elif score < cutoff[1]:
                return "neg"
            else:
                return "neu"
    
