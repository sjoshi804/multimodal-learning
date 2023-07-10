import time
import wandb
import torch
import logging
import torch.distributed as dist
from torch.cuda.amp import autocast
from tqdm import tqdm
from torch.nn import CosineSimilarity
import matplotlib.pyplot as plt 
def pair_analysis(dataloader, options, model):
    vals = []
    epochs = []
    cos = CosineSimilarity(dim = 0, eps = 1e-6)
    for i in range(0, 32, 4):
        if i != 0:
            checkpoint = torch.load("logs/Full Easy 10/checkpoints/epoch_" + str(i) + ".pt", map_location = options.device)
            \
            state_dict = checkpoint["state_dict"]
            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
        counter = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(reversed(list(enumerate(dataloader.dataset)))): 
                input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True) 
                outputs = model(input_ids = input_ids.unsqueeze(0), attention_mask = attention_mask.unsqueeze(0), pixel_values = pixel_values.unsqueeze(0))
                cosval = cos(outputs.image_embeds[0], outputs.text_embeds[0]).item()
                
                
                vals.append(cosval)
                epochs.append(i)
                counter+= 1
                if counter == 10:
                    break
        print("test3")

    plt.scatter(epochs, vals)
    plt.ylabel("Cos Similarity")
    plt.xlabel("Epoch")
    plt.title("Cosine Similarity of Ten Hardest Pairs")
    plt.savefig('analysis/cossimpaireasy.png')