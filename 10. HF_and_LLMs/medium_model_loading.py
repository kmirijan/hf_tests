import torch
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = "microsoft/phi-2"

model = AutoModel.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

ds = load_dataset("jmhessel/newyorker_caption_contest", "explanation")
dataset = ds['train']
dataset_small = dataset.select(range(2))

print('Embedding with Map')
ds_embed = dataset_small.map(
    lambda example: {
        "embeddings": model(
            **tokenizer(example['image_description'],
                       return_tensors="pt", 
                       return_attention_mask=False).to(device)
        ).last_hidden_state.mean(dim=1)[0]
        .detach()
        .cpu()
        .numpy()
    }
)

print(len(ds_embed))