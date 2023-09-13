import os
import random
os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

if(__name__ == "__main__"):   
    f2 = open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/full_data.csv', 'r', encoding='UTF8', newline = '') 
    lines = f2.readlines()
    allvals = list(range(1, len(lines)))
    random.shuffle(allvals)
    with open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/rand_50.csv', 'w', encoding='UTF8', newline = '') as f:
        f.write('caption,file\n')
        for i in allvals[0: int(len(allvals)*0.50)]:
            f.write(lines[i])
            
    