import torch
import pandas as pd

from transformers import MarianMTModel, MarianTokenizer

def load_models(model_name_en_to_fr: str, model_name_fr_to_en, device: str):
    
    """ A function that loads translation models for english to french and french to english

    Args:
        model_name_en_to_fr (str): A model to translate English to French. 
        model_name_fr_to_en (str): A model to translate French to English. 
        device (torch.device): cpu, cuda or mps.

    Returns:
        model, tokenizer, model, tokenizer: returns models and tokenizers for the backtraslation task.
    """
    
    # english to french tokenizer and model
    tokenizer_en_to_fr = MarianTokenizer.from_pretrained(model_name_en_to_fr)
    model_en_to_fr = MarianMTModel.from_pretrained(model_name_en_to_fr).to(device)

    # french to english tokenizer and model
    tokenizer_fr_to_en = MarianTokenizer.from_pretrained(model_name_fr_to_en)
    model_fr_to_en = MarianMTModel.from_pretrained(model_name_fr_to_en).to(device)

    if torch.cuda.device_count() > 1:
        model_en_to_fr = torch.nn.DataParallel(model_en_to_fr)
        model_fr_to_en = torch.nn.DataParallel(model_fr_to_en)
        print(f"Using {torch.cuda.device_count()} GPUs")
        
    return model_en_to_fr, tokenizer_en_to_fr, model_fr_to_en, tokenizer_fr_to_en

def backtranslate(text, model1, tokenizer1, model2, tokenizer2, device):
    
    """ A function that takes a text input and performs backtranslation data augmentation on that.

    Args:
        text (str): A review from one of our datasets
        model1 (model): A model to translate text from English to French.
        tokenizer1 (tokenizer): Tokenizer for eng to fr.
        model2 (model): A model to translate text from French to English.
        tokenizer2 (tokenizer): Tokenizer for fr to eng.

    Returns:
        str: A backtranslated version of our original text.
    """
    
    # translate from englis to french
    inputs = tokenizer1(text, return_tensors = "pt", padding = True, truncation = True).to(device)
    outputs = model1.generate(**inputs)
    translated_text = tokenizer1.decode(outputs[0], skip_special_tokens = True)
    
    # translate back to english
    inputs_back = tokenizer2(translated_text, return_tensors = "pt", padding = True, truncation = True).to(device)
    outputs_back = model2.generate(**inputs_back)
    backtranslated_text = tokenizer2.decode(outputs_back[0], skip_special_tokens = True)
    
    return backtranslated_text

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_en_to_fr, tokenizer_en_to_fr, model_fr_to_en, tokenizer_fr_to_en = load_models(model_name_en_to_fr = "Helsinki-NLP/opus-mt-en-fr", model_name_fr_to_en = "Helsinki-NLP/opus-mt-fr-en", device = device)
    
    # data should be saved in a txt file as label \t sentence format similar to EDA and AEDA
    data_path  = "path to txt file"
    data_aug = []

    with open(data_path, 'r') as train_orig:
        for line in train_orig:
            line1 = line.split('\t')
            label = line1[0]
            sentence = line1[1]
            
            sentence_aug = backtranslate(sentence, model_en_to_fr, tokenizer_en_to_fr, model_fr_to_en, tokenizer_fr_to_en, device)
            line_aug = '\t'.join([label, sentence_aug])
            
            data_aug.append(line_aug)
            data_aug.append(line)
            
    split_data = [row.split('\t') for row in data_aug]

    # convert the list of tuples into a dataframe
    df = pd.DataFrame(split_data, columns=['label', 'review'])

    # convert label column to the appropriate data type if needed
    df['label'] = df['label'].astype(int)
    
    # save augmentated data as a .csv file
    df.to_csv('path to save file', index = False)