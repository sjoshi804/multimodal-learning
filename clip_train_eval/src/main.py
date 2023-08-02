import os
os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

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
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle

from pkgs.openai.clip import load as load_model

from .train import train
from .evaluate import evaluate
from .data import load as load_data
from .data import load_eval as load2, get_revised_train_dataloader
from .parser import parse_args
from .scheduler import cosine_scheduler
from .logger import get_logger, set_logger

mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")

def worker(rank, options, logger):
    options.rank = rank
    options.master = rank == 0
    
    set_logger(rank = rank, logger = logger, distributed = options.distributed)

    if(options.device == "cuda"):
        options.device += ":" + str(options.device_ids[options.rank] if options.distributed else options.device_id)

    logging.info(f"Using {options.device} device")

    if(options.master):
        logging.info("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"{key}: {value}")
                file.write(f"{key}: {value}\n")

    if(options.distributed):
        dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)
    
    options.batch_size = options.batch_size // options.num_devices

    model, processor = load_model(name = options.model_name, pretrained = options.pretrained)

    if(options.device == "cpu"):
        model.float()
    else:
        torch.cuda.set_device(options.device_ids[options.rank] if options.distributed else options.device_id)
        model.to(options.device)
        if(options.distributed):
            model = DDP(model, device_ids = [options.device_ids[options.rank]])
    if(options.full):
        options.eval_data_type = "CIFAR10"
        options.eval_test_data_dir = '/data/cifar10'
        options.eval_train_data_dir = '/data/cifar10'
    data = load_data(options, processor)
    optimizer = None
    scheduler = None
    if(data["train"] is not None):        
        weight_decay_parameters = []
        no_weight_decay_parameters = []

        for name, parameter in model.named_parameters():
            if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                weight_decay_parameters.append(parameter)
                
            if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                no_weight_decay_parameters.append(parameter)

        optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": options.weight_decay}], lr = options.lr, betas = (options.beta1, options.beta2), eps = options.eps)
        scheduler = cosine_scheduler(optimizer, options.lr, options.num_warmup_steps, data["train"].num_batches * options.epochs)

    start_epoch = 0
    if(options.checkpoint is not None):
        if(os.path.isfile(options.checkpoint)):
            checkpoint = torch.load(options.checkpoint, map_location = options.device)
            start_epoch = checkpoint["epoch"]
            state_dict = checkpoint["state_dict"]
            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            if(optimizer is not None): optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"Loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")

    cudnn.benchmark = True
    cudnn.deterministic = False

    if(options.wandb and options.master):
        logging.debug("Starting wandb")
        wandb.init(project = "efficient-multimodal-learning", notes = options.notes, tags = [], config = vars(options))
        wandb.run.name = options.name
        wandb.save(os.path.join(options.log_dir_path, "params.txt"))
    if(options.full and start_epoch == 30):
        logging.info("CIFAR10 Zero Shot")
        evaluate(start_epoch, model, processor, data, options)
        options.linear_probe = True
        data = load2(options, processor, data)
        logging.info("CIFAR10 Linear Probe")
        evaluate(start_epoch, model, processor, data, options)
        options.eval_data_type = "CIFAR100"
        options.eval_test_data_dir = '/data/cifar100'
        options.eval_train_data_dir = '/data/cifar100'
        options.linear_probe = False
        data = load2(options, processor, data)
        logging.info("CIFAR100 Zero Shot")
        evaluate(start_epoch, model, processor, data, options)
        options.linear_probe = True
        data = load2(options, processor, data)
        logging.info("CIFAR100 Linear Probe")
        evaluate(start_epoch, model, processor, data, options)
        options.eval_data_type = "ImageNet1K"
        options.eval_test_data_dir = '/data/ILSVRC/test'
        options.eval_train_data_dir = '/data/ILSVRC/train'
        options.linear_probe = False
        data = load2(options, processor, data)
        logging.info("ImageNet1K Zero Shot")
        evaluate(start_epoch, model, processor, data, options)
        options.linear_probe = True
        data = load2(options, processor, data)
        logging.info("ImageNet1K Linear Probe")

        evaluate(start_epoch, model, processor, data, options)
    else:
        evaluate(start_epoch, model, processor, data, options)

    if(data["train"] is not None):
        options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        os.makedirs(options.checkpoints_dir_path, exist_ok = True)

        scaler = GradScaler()

        best_loss = np.inf
        
        # Curriculum Learning implementation
        curriculum = None
        if options.curriculum != "":
            with open(options.curriculum, "rb") as f:
                curriculum = pickle.load(f)
                
        # Main Training Loop
        for epoch in range(start_epoch + 1, options.epochs + 1):
            if(options.master): 
                logging.info(f"Starting Epoch {epoch}")

            start = time.time()
            
            # Set current epoch training data according to curriculum if specified
            if curriculum is not None:
                data["train"] = get_revised_train_dataloader(data["train"].dataset, curriculum[epoch - 1], options)
                
            train(epoch, model, data, optimizer, scheduler, scaler, options)
            end = time.time()

            if(options.master): 
                logging.info(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")

            metrics = evaluate(epoch, model, processor, data, options)

            if(options.master):
                checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch_{epoch}.pt"))
                if("loss" in metrics):
                    if(metrics["loss"] < best_loss):
                        best_loss = metrics["loss"]
                        torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch.best.pt"))
            
    if(options.distributed):
        dist.destroy_process_group()

    if(options.wandb and options.master):
        wandb.finish()

if(__name__ == "__main__"):    
    options = parse_args()

    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    os.makedirs(options.log_dir_path, exist_ok = True)
    logger, listener = get_logger(options.log_file_path)

    listener.start()

    ngpus = torch.cuda.device_count()
    if(ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if(ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if(options.device_ids is None):
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(worker, nprocs = options.num_devices, args = (options, logger))
    
    listener.stop()
