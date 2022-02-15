class BertSentiment:
    def __init__(self):
        # Load Tokenizer and Config
        tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        config = BertConfig.from_pretrained('indobenchmark/indobert-base-p1')
        config.num_labels = DocumentSentimentDataset.NUM_LABELS

        self.model = BertForSequenceClassification(config=config)
        self.model.load_state_dict(torch.load('bert-model/indonlu_pretrained.pth', map_location=torch.device('cpu')))
        self.model.eval()
    def BERT_sentiment(text, type = "raw"):
        subwords = tokenizer.encode(text)
        subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)
        logits = model(subwords)[0]
        label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
        l = F.softmax(logits, dim=-1).squeeze().tolist()
        return [label_sentiment1((l[0] - l[2])/sum(l))]
    
    
class VaderSentiment:
    def __init__(self):
        import nltk
        import warnings
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import csv
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
        return [label_sentiment2(res['compound'])]

def label_sentiment1(score):
    if score > 0.01:
        return "pos"
    elif score < -0.01:
        return "neg"
    else:
        return "neu"

def BERT_sentiment(text):
  subwords = tokenizer.encode(text)
  subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)
  logits = model(subwords)[0]
  label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
  l = F.softmax(logits, dim=-1).squeeze().tolist()
  return [label_sentiment1((l[0] - l[2])/sum(l))]

def label_sentiment2(score):
    if score > 0.1:
        return "pos"
    elif score < -0.1:
        return "neg"
    else:
        return "neu"
    
