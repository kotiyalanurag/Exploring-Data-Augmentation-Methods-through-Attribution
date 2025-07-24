# training script for mT5 model - language translation task

import numpy as np
import pandas as pd

import ast
import evaluate

from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM

class mT5:
    def __init__(self, checkpoint = "google/mt5-small", batch_size = 16, learning_rate = 5e-4, epochs = 10, experiment = "./mt5-small-finetuned-en-af-10k", source = "en", target = "af"):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.experiment = experiment
        self.source = source
        self.target = target
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.metric = evaluate.load("sacrebleu")

    def load_data(self, train_file, val_file, test_file):
        
        train = pd.read_csv(train_file)
        val = pd.read_csv(val_file)
        test = pd.read_csv(test_file)

        train_dataset = Dataset.from_pandas(train)
        val_dataset = Dataset.from_pandas(val)
        test_dataset = Dataset.from_pandas(test) 

        train_dataset = self.parse_translation_column(train_dataset)
        val_dataset = self.parse_translation_column(val_dataset)
        test_dataset = self.parse_translation_column(test_dataset)

        tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_eval = val_dataset.map(self.preprocess_function, batched=True)
        tokenized_test = test_dataset.map(self.preprocess_function, batched=True)

        return tokenized_train, tokenized_eval, tokenized_test

    def parse_translation_column(self, dataset):
        
        def parse_row(row):
            
            row["translation"] = ast.literal_eval(row["translation"])
            return row
        
        return dataset.map(parse_row)

    def preprocess_function(self, examples):
        
        inputs = [ex[self.source] for ex in examples['translation']]
        targets = [ex[self.target] for ex in examples['translation']]
        
        model_inputs = self.tokenizer(inputs, max_length=64, truncation=True, padding='max_length')
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=64, truncation=True, padding='max_length')
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    def postprocess_text(self, preds, labels):
        
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
    
        return preds, labels

    def compute_metrics(self, eval_preds):
        
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
    
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
        
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, smooth_method="exp")
        result = {"bleu": result["score"]}
    
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def _train(self, train_file, val_file, test_file):
        
        tokenized_train, tokenized_eval, tokenized_test = self.load_data(train_file, val_file, test_file)

        training_args = Seq2SeqTrainingArguments(
            output_dir = self.experiment,
            eval_strategy = "steps",
            learning_rate = self.learning_rate,
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size = self.batch_size,
            weight_decay = 0.01,
            save_total_limit = 3,
            num_train_epochs = self.epochs,
            logging_steps = 250,
            predict_with_generate = True,
            fp16 = False,
            load_best_model_at_end = True,  
            metric_for_best_model = "bleu",
        )

        trainer = Seq2SeqTrainer(
            model = self.model,
            args = training_args,
            train_dataset = tokenized_train,
            eval_dataset = tokenized_eval,
            tokenizer = self.tokenizer,
            compute_metrics = self.compute_metrics,
        )

        trainer.train()

        results = trainer.evaluate(eval_dataset = tokenized_test)

        return results

if __name__ == "__main__":
    
    train_file = "path to train file"
    val_file = "path to val file"
    test_file = "path to test file"
    source = "code for source language i.e., en"
    target = "code for target language i.e., af, cs, cy, es, ro"
    
    mT5_trainer = mT5(source = source, target = target)
    
    results = mT5_trainer._train(train_file, val_file, test_file)
    
    print(results)