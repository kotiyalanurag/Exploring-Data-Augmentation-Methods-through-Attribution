# script to obtain the importance scores for our classification task using layer integrated gradients algorithm
# model dependant method so we need to have a fine tuned model for this to work

import torch
import pandas as pd

from captum.attr import LayerIntegratedGradients
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class LIG:
    def __init__(self, checkpoint = "bert-base-uncased", normalize = True):
        self.checkpoint = checkpoint
        self.normalize = normalize
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(self.device)
        self.layer = self.model.bert.embeddings.word_embeddings
        self.lig = LayerIntegratedGradients(self.predict, self.layer, multiply_by_inputs = True)
        
    def calculate_attributions(self, review, label):

        word_attributions = []

        review_enc = self.tokenizer.encode(review, return_tensors = "pt").to(self.device)

        attr = self.lig.attribute(inputs = review_enc, target = label)

        attributions = self.summarize_attributions(attr)

        decode_text = self.tokenizer.decode(review_enc[0])

        for word, score in zip(decode_text.split(), attributions.tolist()):
            if word != "[CLS]" and word != "[SEP]":
                word_attributions.append((word, score))

        return attributions, word_attributions
    
    def get_attributions(self, input_file):

        train_features, train_labels = self.load_data(input_file = input_file)
        
        attribution_scores = []

        for text, label in zip(train_features, train_labels):

            attributions, _ = self.calculate_attributions(text, label)
            
            attribution_scores.append(sum(attributions.tolist()) / len(attributions.tolist()) if self.normalize else sum(attributions.tolist()))
              
        return attribution_scores
    
    def load_data(self, input_file):
        
        train_dataset = pd.read_csv(input_file)
        
        train_features = train_dataset.review.tolist()
        train_labels = train_dataset.label.tolist()
        
        return train_features, train_labels
    
    def predict(self, inputs, token_type_ids = None, position_ids = None, attention_mask = None):
        
        output = self.model(inputs, token_type_ids = token_type_ids,position_ids = position_ids, attention_mask = attention_mask)
        
        return output.logits
    
    def summarize_attributions(self, attributions):

        attributions = attributions.sum(dim = -1).squeeze(0)

        attributions = attributions / torch.norm(attributions)

        return attributions

if __name__ == "__main__":
    
    input_file = "path to input dataset file"
    output_file = "path to save processed dataset"
    
    lig_attribute = LIG()
    
    attribution_scores = lig_attribute.get_attributions(input_file = input_file)
    
    df = pd.read_csv(input_file)
    df['importance'] = attribution_scores
    df.to_csv(output_file, index = False)  
