import time
import math
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from transformers import get_linear_schedule_with_warmup
from  torch.cuda.amp import autocast

from tqdm import tqdm
import os

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

def train_epochs(model, dataloader, epoch, optimizer, scheduler, args, tb_writer=None):
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

    # model.to(device)
    model.train()
    data_time_m = AverageMeter()
    end = time.time()
    step = 0
    for e in range(epoch):
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            with autocast():
                signals, texts = batch
                assert not torch.isnan(signals).any()
                # signals = signals.to(device=device, non_blocking=True)
                # texts = texts.to(device=device, non_blocking=True)
                input_ids = texts["input_ids"]
                attention_mask = texts["attention_mask"]

                data_time_m.update(time.time() - end)
                optimizer.zero_grad()

                if args.signal_encoder == "xresnet1d101":
                    signals = signals.permute(0, 2, 1)
                loss, caption_loss, contrastive_loss = model(images=signals, text=input_ids, attention_mask=attention_mask, return_loss=True)
                # print(loss)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                accelerator.log({"training_loss": loss}, step=step)
                accelerator.log({"caption_loss": caption_loss}, step=step)
                accelerator.log({"contrastive_loss": contrastive_loss}, step=step)
                step += 1
        accelerator.save_model(model, os.path.join(args.save_path, f"epoch_{e}"))
            # if args.clip_grad > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
        