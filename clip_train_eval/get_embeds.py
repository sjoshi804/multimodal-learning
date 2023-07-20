import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import torch
import clip
import pandas as pd
from tqdm.notebook import tqdm
from pkg_resources import packaging
from PIL import Image 
from tqdm import tqdm 
import torch
from torch.nn import CosineSimilarity
import time

model, preprocess = clip.load("ViT-B/32")

model.eval()
image_embeds = []
text_embeds = []
indexes2 = []
        
with torch.no_grad():
    err_count = 0
    batch_image = []
    batch_text = []
    cos = CosineSimilarity(dim = 1, eps = 1e-6)
    df = pd.read_csv('/home/arnavj/multimodal-learning/clip_train_eval/dsets/full_data.csv', sep=',')
    
    lines = len(df.index)
    counter = 0
    counter2 = 0
    results = {}
    for index, row in tqdm(df.iterrows()):
        try:
            image = preprocess(Image.open(f'{row.file}'))
            text = clip.tokenize(row.caption)
            
        except Exception as e:
            print(str(e))
            df.drop(index, inplace=True)
            continue
        if len(batch_image) < 8192:
            #print(image.size())
            #print(text.size())
            batch_image.append(image.unsqueeze(dim=0))
            batch_text.append(text)

        else:
            
            #print("index_reached", index)
            batch_image = torch.cat(batch_image).to("cuda")
            batch_text = torch.cat(batch_text).to("cuda")
            #print(batch_text.size())
            #print(batch_image.size())
            image_features = model.encode_image(batch_image)
            text_features = model.encode_text(batch_text)
            image_embeds.extend(image_features.tolist())
            text_embeds.extend(text_features.tolist())
            #print(vals)
            del batch_image
            del batch_text
            del image_features
            del text_features
            torch.cuda.empty_cache()
            batch_image = [image.unsqueeze(dim=0)]
            batch_text = [text]
            indexes2.append(index)
    batch_image = torch.cat(batch_image).to("cuda")
    batch_text = torch.cat(batch_text).to("cuda")
    #print(batch_text.size())
    #print(batch_image.size())
    image_features = model.encode_image(batch_image)
    text_features = model.encode_text(batch_text)
    image_embeds.extend(image_features.tolist())
    text_embeds.extend(text_features.tolist())
    #print(vals)
    del batch_image
    del batch_text
    del image_features
    del text_features
    torch.cuda.empty_cache()
    df = df.reset_index(drop=True)
    df.to_csv("/home/arnavj/multimodal-learning/clip_train_eval/dsets/full_data.csv", index=False, sep=',', encoding='utf-8')

    torch.save(text_embeds, "full_text_embeds.pt")
    torch.save(image_embeds, "full_image_embeds.pt")