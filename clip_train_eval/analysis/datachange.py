import os
os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 


if(__name__ == "__main__"):   
     with open('dsets/imagenet_like_train2.csv', 'w', encoding='UTF8', newline = '') as f:
        f2 = open('dsets/imagenet_like_used.csv', 'r', encoding='UTF8', newline = '') 
        f3 = open('dsets/imagenet_like_train.csv', 'r', encoding='UTF8', newline = '')
        lines = f3.readlines()
        lines2 = f2.readlines()
        counter = 0
        for i in lines2:
            f.write(lines[int(i)+1])
            
    