import os
os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import torch
from tqdm import tqdm
from torch.nn import CosineSimilarity

def create_subset(dataloader, percent_subset, options, model):
    cosvalues = []
    cos = CosineSimilarity(dim = 1, eps = 1e-6)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            cosvalues.extend(zip(batch["index"].tolist(),(cos(outputs.image_embeds, outputs.text_embeds)).tolist()))
    cosvalues.sort(key = lambda x : x[1])      
    with open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/imagenet_like_small_test2.csv', 'w', encoding='UTF8', newline = '') as f:
        f2 = open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/imagenet_like_small.csv', 'r', encoding='UTF8', newline = '')
        
        lines = f2.readlines()
        for i, _ in cosvalues:
            f.write(str(_) + " " + lines[i + 1 ])
    
    