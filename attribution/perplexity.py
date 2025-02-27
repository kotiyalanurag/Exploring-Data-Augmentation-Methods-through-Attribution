# script to obtain importance scores for our language translation task using perplexity based text quality filtering
# model agnostic method but we need a PLM to calculate perplexities which defaults to GPT-Neo (125M) model

import ast
import math
import torch
import spacy
import string
import pandas as pd

from torch.utils.data import DataLoader
from transformers import GPTNeoForCausalLM, AutoTokenizer

class Filters:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def has_first_letter_caps(self, sentence):
        return sentence[0].isupper()

    def no_all_caps(self, sentence):
        return not sentence.isupper()

    def terminal_punctuation(self, sentence):
        return sentence.endswith(('.', '!', '?'))

    def word_repetition_ratio_ge_0_2(self, sentence):
        words = sentence.lower().split()
        unique_words = set(words)
        if len(words) == 0:
            return False
        repetition_ratio = (len(words) - len(unique_words)) / len(words)
        return repetition_ratio >= 0.2

    def digit_punctuation_ratio_ge_0_25(self, sentence):
        tokens = sentence.split()
        if len(tokens) == 0:
            return False
        digit_punctuation_count = sum(1 for token in tokens if any(char.isdigit() or char in string.punctuation for char in token))
        return digit_punctuation_count / len(tokens) >= 0.25

    def no_special_characters(self, sentence):
        return not any(char in string.punctuation for char in sentence)

    def stop_word_match_ge_2(self, sentence):
        doc = self.nlp(sentence)
        stop_word_count = sum(token.is_stop for token in doc)
        return stop_word_count >= 2

    def javascript_flag(self, sentence):
        return 'javascript' in sentence.lower()

    def token_count_ge_3(self, sentence):
        doc = self.nlp(sentence)
        return len(doc) >= 3

    def word_count_3_256(self, sentence):
        word_count = len(sentence.split())
        return 3 <= word_count <= 256

    def has_object(self, sentence):
        doc = self.nlp(sentence)
        return any([token.dep_ == 'dobj' for token in doc])

    def has_noun(self, sentence):
        doc = self.nlp(sentence)
        return any([token.pos_ == 'NOUN' for token in doc])

    def has_determiner(self, sentence):
        doc = self.nlp(sentence)
        return any([token.pos_ == 'DET' for token in doc])

    def text_complexity_c1(self, sentence):
        return len(sentence.split()) > 10

class Perplexity(Filters):
    def __init__(self, checkpoint = "EleutherAI/gpt-neo-125M"):
        super().__init__()
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(self.device)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        
    def calculate_perplexity_in_batches(self, sentences, batch_size = 8, max_length = 64):

        self.model.eval()  
        
        total_loss = 0.0
        total_tokens = 0

        dataloader = DataLoader(sentences, batch_size = batch_size)

        for batch in dataloader:
           
            encodings = self.tokenizer(batch, return_tensors = 'pt', padding = True, truncation = True, max_length = max_length)
            input_ids = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)

            with torch.no_grad():
    
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = -100  
                
                outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(1)  
                total_tokens += attention_mask.sum().item()  

        avg_loss = total_loss / total_tokens
        
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
        
    def load_data(self, input_file, source, target, lang_dict):
        
        df = pd.read_csv(input_file)
        
        df["translation"] = df["translation"].apply(ast.literal_eval)
        
        df[f"{lang_dict[source]}"] = df["translation"].apply(lambda x: x.get(f"{source}", ""))
        df[f"{lang_dict[target]}"] = df["translation"].apply(lambda x: x.get(f"{target}", ""))
        
        df['English'] = df['English'].astype(str)
        
        df['first_letter_caps'] = df['English'].apply(lambda text: 1 if self.has_first_letter_caps(text) else 0)
        df['no_all_caps'] = df['English'].apply(lambda text: 1 if self.no_all_caps(text) else 0)
        df['terminal_punctuation'] = df['English'].apply(lambda text: 1 if self.terminal_punctuation(text) else 0)
        df['word_repetition_ratio'] = df['English'].apply(lambda text: 1 if self.word_repetition_ratio_ge_0_2(text) else 0)
        df['digit_punctuation_ratio'] = df['English'].apply(lambda text: 1 if self.digit_punctuation_ratio_ge_0_25(text) else 0)
        df['no_special_characters'] = df['English'].apply(lambda text: 1 if self.no_special_characters(text) else 0)
        df['stop_word_match'] = df['English'].apply(lambda text: 1 if self.stop_word_match_ge_2(text) else 0)
        df['javascript_flag'] = df['English'].apply(lambda text: 1 if self.javascript_flag(text) else 0)
        df['token_count'] = df['English'].apply(lambda text: 1 if self.token_count_ge_3(text) else 0)
        df['word_count_range'] = df['English'].apply(lambda text: 1 if self.word_count_3_256(text) else 0)
        df['has_object'] = df['English'].apply(lambda text: 1 if self.has_object(text) else 0)
        df['has_noun'] = df['English'].apply(lambda text: 1 if self.has_noun(text) else 0)
        df['has_determiner'] = df['English'].apply(lambda text: 1 if self.has_determiner(text) else 0)
        df['text_complexity_c1'] = df['English'].apply(lambda text: 1 if self.text_complexity_c1(text) else 0)
        
        return df
    
    def calculate_filter_weights(self, input_file, source, target, lang_dict):
        
        df = self.load_data(input_file = input_file, source = source, target = target, lang_dict = lang_dict)
        
        original_sentences = df['English'].tolist()
        
        original_ppl = self.calculate_perplexity_in_batches(original_sentences)
        
        filters = ['first_letter_caps', 'no_all_caps', 'terminal_punctuation', 'word_repetition_ratio', 'digit_punctuation_ratio', 'no_special_characters', 'stop_word_match', 'javascript_flag', 'token_count', 'word_count_range', 'has_object', 'has_noun', 'has_determiner', 'text_complexity_c1']
        
        filter_weights = {}
        
        for flt in filters:
            
            subset_sentences = df[df[f"{flt}"] == 1]['English'].tolist()
            
            subset_ppl = self.calculate_perplexity_in_batches(subset_sentences)
            
            if subset_ppl > original_ppl or math.isnan(subset_ppl):
                
                weight = 0.0
                
            else:
                
                weight = (original_ppl - subset_ppl) / original_ppl
                
            filter_weights[f'{flt}'] = weight
            
        return df, filter_weights
    
    def process_data(self, input_file, source, target, lang_dict):
    
        df, filter_weights = self.calculate_filter_weights(input_file = input_file, source = source, target = target, lang_dict = lang_dict)
        
        sum_filter_values = sum(filter_weights.values())

        for filter_name, weight in filter_weights.items():
            
            df[f'weighted_{filter_name}'] = df[filter_name] * weight

        df['score'] = df[[f'weighted_{filter_name}' for filter_name in filter_weights]].sum(axis=1) / sum_filter_values
        
        df["translation"] = df.apply(lambda row: {f"{source}": row[f"{lang_dict[source]}"], f"{target}": row[f"{lang_dict[target]}"]}, axis = 1)
        
        df = df[["translation", "score"]]  
        
        df.rename(columns={"score": "importance"}, inplace = True)
        
        return df, filter_weights
    
if __name__ == "__main__":
    
    input_file = "path to input dataset file"
    output_file = "path to save processed dataset"
    
    source = "code for source language i.e., en"
    
    target = "code for target language i.e., af, cs, cy, es, ro"
    
    lang_dict = {"en": "English", "af": "Afrikaans", "cs": "Czech", "cy": "Welsh", "es": "Spanish", "ro": "Romanian"}
    
    perplexity = Perplexity()
    
    df, filter_weights = perplexity.process_data(input_file = input_file, source = source, target = target, lang_dict = lang_dict)
    df.to_csv(output_file, index = False)