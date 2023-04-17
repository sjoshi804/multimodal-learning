import os
os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 


if(__name__ == "__main__"):   
     with open('dsets/cifar10_hard_80.csv', 'w', encoding='UTF8', newline = '') as f:
        f2 = open('analysis/datasorted', 'r', encoding='UTF8', newline = '') 
        lines = f2.readlines()
        counter = 0
        for i in range(0, int(len(lines)*.8)):
            f.write(str(counter) + ',' + lines[i][lines[i].index(',') + 1:])
            counter +=1
            
    