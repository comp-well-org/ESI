import time
import math
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from transformers import get_linear_schedule_with_warmup
from  torch.cuda.amp import autocast

from tqdm import tqdm

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def export_esi(model, dataloader, epoch, optimizer, scheduler, args, tb_writer=None):
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], 
                              log_with="wandb",
                              project_dir=args.save_path)
    accelerator.init_trackers(
        project_name="esi", 
        config={"dropout": 0.1, "learning_rate": args.lr},
    )   
    device = accelerator.device
    
    num_training_steps = args.epochs * len(dataloader)
    warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(
          model, optimizer, dataloader, scheduler
      )

    accelerator.load_state(args.save_path)
    accelerator.save_model(model, "./saved_models/convnext_base/")
        