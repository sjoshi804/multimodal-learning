import os, psutil

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import sys
import time
import wandb
import torch
import logging
import warnings
import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from pkgs.openai.clip import load as load_model
import numpy
import inspect
import pickle

class ImageCaptionDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor, inmodal = False):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path, sep = delimiter)
        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions = processor.process_text(df[caption_key].tolist())
        self.processor = processor
        
        self.inmodal = inmodal
        
        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        
        
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.process_image(Image.open(os.path.join(self.root, self.images[idx])))
        
        return item, idx

checkpoint_full = torch.load("/home/arnavj/multimodal-learning/clip_train_eval/logs/full/checkpoints/epoch_30.pt", map_location = 'cuda')
state_dict_full = checkpoint_full["state_dict"]
if(next(iter(state_dict_full.items()))[0].startswith("module")):
    state_dict_full = {key[len("module."):]: value for key, value in state_dict_full.items()}

batchsize = 512

model, processor = load_model(name = "RN50", pretrained = False)
torch.cuda.set_device(0)
model.to('cuda')
model.load_state_dict(state_dict_full)

print("Loaded Checkpoints")
dataset = ImageCaptionDataset("/home/arnavj/multimodal-learning/clip_train_eval/dsets/full_data.csv", image_key = 'file', caption_key = 'caption', delimiter = ',', processor = processor, inmodal = False)

dataloader = DataLoader(dataset, batch_size = batchsize, shuffle = False, num_workers = 8, pin_memory = True, sampler = None, drop_last = False)
dataloader.num_samples = len(dataloader) * batchsize 
dataloader.num_batches = len(dataloader)
#images = torch.load("image_embeds2.pt")
#text = torch.load("text_embeds2.pt")
vals = []
block_size = batchsize
model.eval()
coses = []
#print([method_name for method_name in dir(model)
#                  if callable(getattr(model, method_name))])
process = psutil.Process()
with torch.no_grad():
    for batch, idx in tqdm(dataloader):
        #image_batch = batch['pixel_values'].to("cuda")
        #text_batch = text[i*block_size:(i+1)*block_size].to("cuda")
        #print(image_batch)
        image_features = model.get_image_features(batch['pixel_values'].to("cuda")).float()
        text_features = model.get_text_features(input_ids = batch['input_ids'].to("cuda"), attention_mask = batch['attention_mask'].to("cuda")).float()

        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        #outputs = model(image = image_batch, text = text_batch)
        #outputs2 = model2(image = image_batch, text = text_batch)
        cos = numpy.diagonal(similarity, 0)
        #vals.extend(list(zip(cos, idx)))
        coses.extend(cos)
        del cos
        del image_features
        del text_features
        del similarity
        #print(process.memory_info().rss)

#vals.sort(reverse=True)
curr_file = open('full_trained_sort.pickle', 'wb')
pickle.dump(coses, curr_file)
#curr_file2 = open('full_trained_sort2.pickle', 'wb')
#pickle.dump(vals, curr_file2)




