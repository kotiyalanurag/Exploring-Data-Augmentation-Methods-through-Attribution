import torch
import pandas as pd
import transformers 

from huggingface_hub import login

def load_model(model_id):
    
    """ A function that loads our llama-3.1-8B-Instruct model using transformer pipeline for generation

    Args:
        model_id (str): Name of model to load from Hugging Face.

    Returns:
        transformers.pipeline: Return the pipeline object to paraphrase text data
    """
    
    pipeline = transformers.pipeline(
        "text-generation",
        model = model_id,
        model_kwargs = {"torch_dtype": torch.bfloat16},
        device_map = "auto",
        )
    
    return pipeline

def paraphrase(text, pipeline):
    
    """ A function that used default pipeline and Llama Instruct's capabilities for data augmentation through paraphrasing.

    Args:
        text (str): A review from one of our datasets.
        pipeline (transformers.pipeline): A pipeline object to paraphrase text data.

    Returns:
        str: A paraphrased version of original data i.e., augmented data.
    """
    
    messages = [
    {"role": "system", "content": "You are a helpful chatbot who paraphrases given text."},
    {"role": "user", "content": text},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens = 256,
    )
    
    return outputs[0]["generated_text"][-1]["content"]

if __name__ == "__main__":
    
    # need hugging face token to login as well as permission from repository owners to use Llama 3.1 models
    login("place your hugging face token for Llama usage here")
    
    # initializing model pipeline for text generation
    pipeline = load_model(model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    # data should be saved in a txt file as label \t sentence format similar to EDA and AEDA
    data_path  = "path to txt file"
    data_aug = []

    with open(data_path, 'r') as train_orig:
        for line in train_orig:
            line1 = line.split('\t')
            label = line1[0]
            sentence = line1[1]
            
            sentence_aug = paraphrase(sentence)
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