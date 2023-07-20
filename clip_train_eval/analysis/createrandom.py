import os
import random
os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

if(__name__ == "__main__"):   
    f2 = open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/full_data.csv', 'r', encoding='UTF8', newline = '') 
    lines = f2.readlines()
    allvals = list(range(1, len(lines)))
    random.shuffle(allvals)
    with open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/rand1M.csv', 'w', encoding='UTF8', newline = '') as f:
       
        
        for i in allvals[0: 1000000]:
            f.write(lines[i])
            
    