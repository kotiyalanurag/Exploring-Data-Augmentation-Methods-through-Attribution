# training script for BERT model - classification task

import re
import evaluate

import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer

class BERT:
    def __init__(self, checkpoint = "bert-base-uncased", batch_size = 16, learning_rate = 5e-5, epochs = 10, experiment = "./bert-base-finetuned-sst2", dataset = "SST2"):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.experiment = experiment
        self.dataset = dataset
        self.label2id, self.id2label, self.num_labels = self.get_model_config()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = self.num_labels, label2id = self.label2id, id2label = self.id2label)
        self.metric = evaluate.load("accuracy")
    
    def get_model_config(self):
        
        if self.dataset == "SST2":  
            label2id = {"Negative": 0, "Positive": 1}
            id2label = {0: "Negative", 1: "Positive"}
            num_labels = 2
        
        elif self.dataset == "SST5":   
            label2id = {"Very Negative": 0, "Negative": 1, "Neutral": 2, "Positive": 3, "Very Positive": 4}
            id2label = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
            num_labels = 5
        
        elif self.dataset == "IMDB":    
            label2id = {"Negative": 0, "Positive": 1}
            id2label = {0: "Negative", 1: "Positive"}
            num_labels = 2
        
        elif self.dataset == "AGNEWS":   
            label2id = {"Worlds": 0, "Sports": 1, "Business": 2, "Science": 3}
            id2label = {0: "Worlds", 1: "Sports", 2: "Business", 3: "Science"}
            num_labels = 4
            
        elif self.dataset == "IRONY":
            label2id = {"Not Ironic": 0, "Ironic": 1}
            id2label = {0: "Not Ironic", 1: "Ironic"}
            num_labels = 2
        
        else:
            return 0, 0, 0 
            
        return label2id, id2label, num_labels

    def load_data(self, train_file, val_file, test_file):
        
        train = pd.read_csv(train_file)
        dev = pd.read_csv(val_file)
        test = pd.read_csv(test_file)

        train['text'] = train.apply(lambda x: self.clean_text(x['text']), axis = 1)
        test['text'] = test.apply(lambda x: self.clean_text(x['text']), axis = 1)
        dev['text'] = dev.apply(lambda x: self.clean_text(x['text']), axis = 1)
        
        train['text'] = train['text'].astype("string")
        test['text'] = test['text'].astype("string")
        dev['text'] = dev['text'].astype("string")

        train_dataset = Dataset.from_pandas(train)
        test_dataset = Dataset.from_pandas(test)
        dev_dataset = Dataset.from_pandas(dev)

        tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_eval = dev_dataset.map(self.preprocess_function, batched=True)
        tokenized_test = test_dataset.map(self.preprocess_function, batched=True)

        return tokenized_train, tokenized_eval, tokenized_test
    
    def clean_text(self, text):
    
        text = text.lower() 
        text = text.replace(" 's", " is")
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        text = text.replace("'", "")
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
        text = text.strip()
        
        return text

    def compute_metrics(self, eval_pred):
        
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis = -1)
        
        return self.metric.compute(predictions = predictions, references = labels)


    def preprocess_function(self, examples):
        
        return self.tokenizer(examples["review"], truncation = True)

    def _train(self, train_file, val_file, test_file):

        tokenized_train, tokenized_eval, tokenized_test = self.load_data(train_file, val_file, test_file)

        data_collator = DataCollatorWithPadding(self.tokenizer)

        training_args = TrainingArguments(
            output_dir = self.experiment,
            num_train_epochs = self.epochs,
            learning_rate = self.learning_rate,
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size = self.batch_size,
            warmup_steps = 500,
            weight_decay = 0.01,
            logging_dir = '/bertlogs',
            logging_steps = 10,
            eval_strategy = "epoch",
        )
        
        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = tokenized_train,
            eval_dataset = tokenized_eval,
            data_collator = data_collator,
            compute_metrics = self.compute_metrics,
        )
        
        trainer.train()
        
        results = trainer.evaluate(eval_dataset = tokenized_test)
        
        return results
    
if __name__ == "__main__":
    
    train_file = "path to train file"
    val_file = "path to val file"
    test_file = "path to test file"
    dataset = "name of the classification dataset i.e., SST2, SST5, IMDB, AGNEWS, IRONY"
    
    bert_trainer = BERT(dataset = dataset)
    
    results = bert_trainer._train(train_file, val_file, test_file)
    
    print(results)