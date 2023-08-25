import wandb
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  
from pkgs.openai.clip import load as load_model
import pickle
import clip
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

model, preprocess = clip.load('ViT-B/32', 'cuda')
config = eval(open("/data/ILSVRC/test/classes.py", "r").read())
classes, templates = config["classes"], config["templates"]
X1 = torch.load('image_embeds2.pt')
model.eval()

with torch.no_grad():
    zeroshot_weights = []
    for classname in tqdm(classes):
        texts = [template(classname) for template in templates]
        texts = clip.tokenize(texts).cuda() #tokenize
        class_embeddings = model.encode_text(texts) #embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    text_embeddings = torch.stack(zeroshot_weights, dim=1).cuda()
'''
with torch.no_grad():
        text_embeddings = []
        for c in tqdm(classes):
            text = [template(c) for template in templates]
            text_inputs = torch.cat([clip.tokenize(text) for c in cifar100.classes]).to(device)

            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to('cuda'), text_tokens["attention_mask"].to('cuda') 
            text_embedding = umodel.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            text_embedding = text_embedding.mean(dim = 0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 1).to('cuda')
'''
block_size = 256
class_pred = {}
for i in range(1000):
    class_pred[i] = []
with torch.no_grad():
    for i in tqdm(range(X1.shape[0] // block_size + 1)):
        similarity_matrices_i = []
        e = X1[i*block_size:(i+1)*block_size]
        image = e.cuda()
        image /= image.norm(dim = -1, keepdim = True)
        image = image.float()
        text_embeddings = text_embeddings.float()
        logits = (image @ text_embeddings)
        prediction = torch.argmax(logits, dim = 1)
        for x in range(len(prediction)):
            y = prediction[x].item()
            class_pred[y].append(i * block_size + x)

partitions2 = open('/home/arnavj/multimodal-learning/clip_train_eval/analysis/predict_zero_imagenet_like.pickle', 'wb')

pickle.dump(class_pred, partitions2)
        
        