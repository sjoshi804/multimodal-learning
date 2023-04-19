import os
os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import torch
from torch.nn import CosineSimilarity
from tqdm import tqdm    

def calc_cos(dataloader, model, options):
    cosvalues = []
    cos = CosineSimilarity(dim = 0, eps = 1e-6)
    with torch.no_grad():
        for batch_idx, (batch,index) in tqdm(enumerate(dataloader.dataset)): 
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True) 
            outputs = model(input_ids = input_ids.unsqueeze(0), attention_mask = attention_mask.unsqueeze(0), pixel_values = pixel_values.unsqueeze(0))
            cosvalues.append([index, cos(outputs.image_embeds[0], outputs.text_embeds[0]).item()])
    cosvalues.sort(key = lambda x : x[1])      
    