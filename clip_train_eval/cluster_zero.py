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

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model, preprocess = clip.load('ViT-B/32', 'cuda')
model2, processor = load_model(name ='RN50', pretrained = False)
checkpoint_full = torch.load("/home/arnavj/multimodal-learning/clip_train_eval/logs/full/checkpoints/epoch_30.pt", map_location = 'cuda')
state_dict_full = checkpoint_full["state_dict"]
if(next(iter(state_dict_full.items()))[0].startswith("module")):
    state_dict_full = {key[len("module."):]: value for key, value in state_dict_full.items()}
model2.to('cuda')
model2.load_state_dict(state_dict_full)
config = eval(open("/data/ILSVRC/test/classes.py", "r").read())
classes, superclasses, templates = config["classes"], config["superclasses"], config["templates"]
X1 = torch.load('full_model_image_embeds.pt')
print(len(X1))
model.eval()
conversion = {}
for i in range(len(superclasses)):
    for x in superclasses[i]:
        conversion[x] = i
'''
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

            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to('cuda'), text_tokens["attention_mask"].to('cuda') 
            text_embedding = model2.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            text_embedding = text_embedding.mean(dim = 0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 1).to('cuda')

block_size = 256
class_pred = {}

for i in range(len(classes)):
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
            #class_pred[conversion[classes[y]]].append(i * block_size + x)

partitions2 = open('/home/arnavj/multimodal-learning/clip_train_eval/analysis/full_model_partition.pickle', 'wb')
#print((list(class_pred.keys())))

pickle.dump(class_pred, partitions2)
        
        