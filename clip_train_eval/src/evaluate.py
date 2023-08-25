import wandb
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm    
from .scheduler import cosine_scheduler

def get_validation_metrics(model, dataloader, options):
    logging.info("Started validating")

    metrics = {}

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction = "sum").to(options.device)

    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True) 
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            
            umodel = model.module if(options.distributed) else model

            logits_per_image = umodel.logit_scale.exp() * outputs.image_embeds @ outputs.text_embeds.t()
            logits_per_text = logits_per_image.t()

            target = torch.arange(len(input_ids)).long().to(options.device, non_blocking = True)
            loss = (criterion(logits_per_image, target) + criterion(logits_per_text, target)) / 2

            losses.append(loss)

        loss = sum(losses) / dataloader.num_samples
        metrics["loss"] = loss

    logging.info("Finished validating")

    return metrics

def get_zeroshot_metrics(model, processor, test_dataloader, options):
    logging.info("Started zeroshot testing")
    if(options.eval_data_type == "Caltech101"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR100"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "DTD"):
        output_dim = 47
        metric = "accuracy"
    elif(options.eval_data_type == "FGVCAircraft"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "Flowers102"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "Food101"):
        output_dim = 101
        metric = "accuracy"
    elif(options.eval_data_type == "GTSRB"):
        output_dim = 43
        metric = "accuracy"
    elif(options.eval_data_type == "ImageNet1K"):
        output_dim = 1000
        metric = "accuracy"
    elif(options.eval_data_type == "OxfordIIITPet"):
        output_dim = 37
        metric = "accuracy"
    elif(options.eval_data_type == "RenderedSST2"):
        output_dim = 2
        metric = "accuracy"
    elif(options.eval_data_type == "StanfordCars"):
        output_dim = 196
        metric = "accuracy"
    elif(options.eval_data_type == "STL10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "SVHN"):
        output_dim = 10
        metric = "accuracy"

    model.eval()
    umodel = model.module if(options.distributed) else model

    config = eval(open(f"{options.eval_test_data_dir}/classes.py", "r").read())
    classes, templates = config["classes"], config["templates"]
    if options.class_subset:
        sub_class = [134, 254, 288, 291, 292, 308, 309, 312, 323, 341, 414, 417, 425, 429, 430, 435, 456, 460, 463, 468, 470, 472, 476, 483, 487, 492, 494, 497, 498, 506, 524, 525, 531, 535, 537, 552, 577, 578, 579, 581, 605, 619, 625, 642, 649, 654, 667, 669, 672, 678, 697, 702, 704, 705, 713, 717, 720, 725, 732, 737, 749, 750, 753, 760, 766, 777, 781, 786, 794, 804, 816, 818, 824, 833, 836, 838, 846, 850, 865, 867, 878, 882, 888, 889, 893, 907, 915, 916, 922, 927, 950, 953, 957, 962, 965, 967, 971, 972, 975, 977, 978, 981, 986, 988, 997]
        classes = [x for idx, x in enumerate(classes) if idx in sub_class]
    with torch.no_grad():
        text_embeddings = []
        for c in tqdm(classes):
            text = [template(c) for template in templates]
            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.device), text_tokens["attention_mask"].to(options.device) 
            text_embedding = umodel.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            text_embedding = text_embedding.mean(dim = 0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 1).to(options.device)
    if(options.predict):
        predict_matrix = np.zeros((output_dim, output_dim))
        perclass_tot = torch.zeros(output_dim)
    if(options.classes):
        perclass_tot = torch.zeros(output_dim)
        perclass_corr = torch.zeroes(output_dim)
    with torch.no_grad():
        topk = [1, 3, 5, 10]
        correct = {k: 0 for k in topk}
        
        for image, label in tqdm(test_dataloader):
            image, label = image.to(options.device), label.to(options.device)
            image_embedding = umodel.get_image_features(image)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)
            logits = (image_embedding @ text_embeddings)
            ranks = logits.topk(max(topk), 1)[1].T
            predictions = ranks == label

            for k in topk:
                correct[k] += torch.sum(torch.any(predictions[:k], dim = 0)).item() 
            if options.predict:
                predictions = predictions.to('cpu')
                label = label.to('cpu')
                for i in range(len(label)):
                    predict_matrix[label[i]][ranks[0][i]] += 1
                    perclass_tot[label[i]]+=1
            if(options.classes):
                predictions = predictions.to('cpu')
                label = label.to('cpu')
                for i in range(len(label)):
                    perclass_corr[label[i]] += label[i] == ranks[0][i]
                    perclass_tot[label[i]]+=1
    if(options.predict):
        class_10 =  ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        import matplotlib.pyplot as plt
        vals = np.count_nonzero(predict_matrix, axis=1)
        for i in range(10):
            plt.bar(class_10, np.divide(predict_matrix[i], perclass_tot[i]))
            plt.xlabel('Class Names')
            plt.xticks(
                rotation=45, 
                horizontalalignment='right',
                fontweight='light',
                fontsize='small'  
            )
            plt.ylabel('Percentage of Examples placed in Class')
            plt.title('Distribution of Examples for Class ' + class_10[i] + ": Easy Zero Shot" )
            plt.savefig("predict-easy/cifar10_zero_" + class_10[i] + ".png")
            plt.clf()
    if(options.classes != None):
        with open(options.classes, 'w', encoding='UTF8', newline = '') as f:
            for i in range(len(perclass_corr)):
                f.write(str((perclass_corr[i]/perclass_tot[i]).item()) + "\n")
    results = {f"zeroshot_top{k}": correct[k] / test_dataloader.num_samples for k in topk}
    logging.info("Finished zeroshot testing")

    return results

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    
def get_linear_probe_metrics(model, train_dataloader, test_dataloader, options):
    logging.info("Started linear probe testing")
    logging.info(f"Number of train examples: {train_dataloader.num_samples}")
    logging.info(f"Number of test examples: {test_dataloader.num_samples}")

    model.eval()
    umodel = model.module if(options.distributed) else model
    
    images = None
    labels = None
    with torch.no_grad():
        for image, label in tqdm(train_dataloader):
            image = umodel.get_image_features(image.to(options.device)).cpu()
            images = torch.cat([images, image], dim = 0) if(images is not None) else image
            labels = torch.cat([labels, label], dim = 0) if(labels is not None) else label

    train_dataset = torch.utils.data.TensorDataset(images, labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = options.batch_size, shuffle = True)
    
    input_dim = umodel.text_projection.shape[1]
    
    if(options.eval_data_type == "Caltech101"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "CIFAR100"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "DTD"):
        output_dim = 47
        metric = "accuracy"
    elif(options.eval_data_type == "FGVCAircraft"):
        output_dim = 100
        metric = "accuracy"
    elif(options.eval_data_type == "Flowers102"):
        output_dim = 102
        metric = "accuracy"
    elif(options.eval_data_type == "Food101"):
        output_dim = 101
        metric = "accuracy"
    elif(options.eval_data_type == "GTSRB"):
        output_dim = 43
        metric = "accuracy"
    elif(options.eval_data_type == "ImageNet1K"):
        output_dim = 1000
        metric = "accuracy"
    elif(options.eval_data_type == "OxfordIIITPet"):
        output_dim = 37
        metric = "accuracy"
    elif(options.eval_data_type == "RenderedSST2"):
        output_dim = 2
        metric = "accuracy"
    elif(options.eval_data_type == "StanfordCars"):
        output_dim = 196
        metric = "accuracy"
    elif(options.eval_data_type == "STL10"):
        output_dim = 10
        metric = "accuracy"
    elif(options.eval_data_type == "SVHN"):
        output_dim = 10
        metric = "accuracy"

    classifier = LogisticRegression(input_dim = input_dim, output_dim = output_dim).to(options.device)
    optimizer = optim.AdamW([{"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" in name) and parameter.requires_grad)], "weight_decay": 0}, {"params": [parameter for name, parameter in classifier.named_parameters() if(("bias" not in name) and parameter.requires_grad)], "weight_decay": 0.01}])
    scheduler = cosine_scheduler(optimizer, 0.005, 0, len(train_dataloader) * options.linear_probe_num_epochs)
    criterion = nn.CrossEntropyLoss().to(options.device)
    perclass_corr = torch.zeros(output_dim)
    perclass_tot = torch.zeros(output_dim)
    import pickle
    #partition_file = open('/home/arnavj/multimodal-learning/clip_train_eval/analysis/rand_class_size', 'rb')
    #partition = pickle.load(partition_file)
    #partition_file2 = open('/home/arnavj/multimodal-learning/clip_train_eval/analysis/partition_imagenet_like', 'rb')
    #partition2 = pickle.load(partition_file2)
    #per_class = open('/home/arnavj/multimodal-learning/clip_train_eval/analysis/per_class', 'w', encoding='UTF8', newline = '')
    pbar = tqdm(range(options.linear_probe_num_epochs))
    for epoch in pbar:
        cbar = tqdm(train_dataloader, leave = False)
        for index, (image, label) in enumerate(cbar):
            step = len(train_dataloader) * epoch + index
            scheduler(step)
            image, label = image.to(options.device), label.to(options.device)
            logit = classifier(image)
            optimizer.zero_grad()
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            cbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
        pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})

    classifier.eval()
    predict_matrix = np.zeros((output_dim, output_dim))
    with torch.no_grad():
        if(metric == "accuracy"):
            correct = 0
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(umodel.get_image_features(image))
                prediction = torch.argmax(logits, dim = 1)
                correct += torch.sum(prediction == label).item()
                label = label.to('cpu')
                prediction = prediction.to('cpu')
                if options.predict:
                    for i in range(len(label)):
                        predict_matrix[label[i]][prediction[i]] += 1
                        perclass_tot[label[i]] += 1
                if options.classes:    
                    for i in range(len(label)):
                        perclass_corr[label[i]] += (prediction[i] == label[i])
                        perclass_tot[label[i]] += 1
            results = {f"linear_probe_accuracy": correct / test_dataloader.num_samples}
        else:
            correct = torch.zeros(output_dim).to(options.device)
            total = torch.zeros(output_dim).to(options.device)
            for image, label in tqdm(test_dataloader):
                image, label = image.to(options.device), label.to(options.device)
                logits = classifier(umodel.get_image_features(image))
                predictions = torch.argmax(logits, dim = 1)
                
                temp = torch.zeros(output_dim, len(label)).to(options.device)
                temp[label, torch.arange(len(label))] = (predictions == label).float()
                correct += temp.sum(1)
                temp[label, torch.arange(len(label))] = 1                
                total += temp.sum(1)

            results = {f"linear_probe_mean_per_class": (correct / total).mean().cpu().item()}
    
    if(options.predict):
        class_10 =  ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        import matplotlib.pyplot as plt
        vals = np.count_nonzero(predict_matrix, axis=1)
        for i in range(10):
            plt.bar(class_10, np.divide(predict_matrix[i], perclass_tot[i]))
            plt.xlabel('Class Names')
            plt.xticks(
                rotation=45, 
                horizontalalignment='right',
                fontweight='light',
                fontsize='small'  
            )
            plt.ylabel('Percentage of Examples placed in Class')
            plt.title('Distribution of Examples for Class ' + class_10[i] + ": Easy Linear Probe" )
            plt.savefig("predict-easy/cifar10_linear_" + class_10[i] + ".png")
            plt.clf()
    if(options.classes != None):
       
        
        #to_sort = []
        #for i in range(len(perclass_corr)):
        #    to_sort.append(((partition[i]), str((perclass_corr[i]/perclass_tot[i]).item()), str(perclass_tot[i]), list(partition2.keys())[i]))
        #to_sort.sort()
        #for i in to_sort:    
        #    per_class.write(str(i[0]) + " " + i[1] + " " + i[2] + " " + i[3] + "\n")
        #plt.xlim(0,10)
        #plt.savefig('perclass_acc_2')
        with open(options.classes, 'w', encoding='UTF8', newline = '') as f:
            for i in range(len(perclass_corr)):
                f.write(str((perclass_corr[i]/perclass_tot[i]).item()) + "\n")

    
    logging.info("Finished linear probe testing")

    return results

def evaluate(epoch, model, processor, data, options):
    metrics = {}
    
    if(options.master):
        if(data["validation"] is not None or data["eval_test"] is not None):
            if(epoch == 0):
                logging.info(f"Base evaluation")
            else:
                logging.info(f"Epoch {epoch} evaluation")

        if(data["validation"] is not None): 
            metrics.update(get_validation_metrics(model, data["validation"], options))
            
        if(data["eval_test"] is not None): 
            if(data["eval_train"] is not None):
                metrics.update(get_linear_probe_metrics(model, data["eval_train"], data["eval_test"], options))
            else:
                metrics.update(get_zeroshot_metrics(model, processor, data["eval_test"], options))
        
        if(metrics):
            logging.info("Results")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")

            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"evaluation/{key}": value, "epoch": epoch})

    return metrics