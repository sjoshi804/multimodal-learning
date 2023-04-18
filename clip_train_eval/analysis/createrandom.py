import os
import random
os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

if(__name__ == "__main__"):   
     with open('dsets/imagenet_like_random1.csv', 'w', encoding='UTF8', newline = '') as f:
        f2 = open('dsets/imagenet_like_train.csv', 'r', encoding='UTF8', newline = '') 
        lines = f2.readlines()
        allvals = list(range(1, len(lines)))
        random.shuffle(allvals)
        for i in allvals[0: int(len(allvals) * 0.2)]:
            f.write(lines[i])
            
    