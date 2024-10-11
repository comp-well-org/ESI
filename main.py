import torch
from model.convnextv2 import ConvNeXtV2, convnextv2_base, convnextv2_atto, convnextv2_atto, convnextv2_nano, convnextv2_tiny, convnextv2_large
from model.xresnet1d import xresnet1d101
from model.esi import ESI

from utils.ecg_loader import tokenize_collate_fn

from transformers import AutoTokenizer, AutoModel
from utils.ecg_loader import get_ecg_loader
from train import train_epochs

import argparse

def main(args):
    if args.signal_encoder == "convnextv2_base":
        signal_encoder = convnextv2_base(in_chans = args.in_chans, num_classes = args.num_classes, return_embedding=True)
        signal_dim = 1024
    elif args.signal_encoder == "convnextv2_atto":
        signal_encoder = convnextv2_atto(in_chans = args.in_chans, num_classes = args.num_classes, return_embedding=True)
        signal_dim = 320
    elif args.signal_encoder == "convnextv2_nano":
        signal_encoder = convnextv2_nano(in_chans = args.in_chans, num_classes = args.num_classes, return_embedding=True)
        signal_dim = 640
    elif args.signal_encoder == "convnextv2_tiny":
        signal_encoder = convnextv2_tiny(in_chans = args.in_chans, num_classes = args.num_classes, return_embedding=True)
        signal_dim = 768
    elif args.signal_encoder == "convnextv2_large":
        signal_encoder = convnextv2_large(in_chans = args.in_chans, num_classes = args.num_classes, return_embedding=True)
        signal_dim = 1536
    elif args.signal_encoder == "xresnet1d101":
        signal_encoder = xresnet1d101(num_classes=args.num_classes,input_channels=args.in_chans,kernel_size=5,ps_head=0.5,lin_ftrs_head=[128])
        signal_dim = 512
    else:
        raise ValueError("Invalid signal encoder")
    
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    total_tokens = len(tokenizer.vocab)
    
    def wrapper_collate_fn(batch):
        return tokenize_collate_fn(batch, tokenizer)
    
    esi = ESI(
        dim = args.dim,
        image_dim = signal_dim,
        num_tokens = total_tokens,
        pretrained_text_encoder = text_encoder,
        unimodal_depth = 6,            # depth of the unimodal transformer
        multimodal_depth = 6,          # depth of the multimodal transformer
        dim_head=64,
        heads=8,
        ff_mult=4,
        img_encoder=signal_encoder,
        caption_loss_weight=1.,
        contrastive_loss_weight=1.,
        )
    
    trainloader = get_ecg_loader(args.meta_path, args.file_path, args.batch_size, wrapper_collate_fn, args.num_workers)
    optimizer = torch.optim.AdamW(esi.parameters(), lr=args.lr)
    train_epochs(esi, trainloader, args.epochs, optimizer, None, args)
    
if __name__ == '__main__':
    print("Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    
    # data loader
    parser.add_argument('--meta_path', type=str, default='/rdf/data/physio_clip/ecg/meta_data_batch1.csv', help='meta path')
    parser.add_argument('--file_path', type=str, default='/rdf/data/physio_clip/ecg/files', help='file path')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    
    # model
    parser.add_argument('--model', type=str, default='esi', help='model')
    parser.add_argument('--dim', type=int, default=768, help='model dimension')
    parser.add_argument('--signal_encoder', type=str, default='xresnet1d101', help='signal encoder')
    parser.add_argument('--signal_dim', type=int, default=640, help='signal dimension')
    parser.add_argument('--text_dim', type=int, default=768, help='text dimension')
    parser.add_argument('--num_tokens', type=int, default=20000, help='number of text tokens')
    parser.add_argument('--unimodal_depth', type=int, default=6, help='depth of the unimodal transformer')
    parser.add_argument('--multimodal_depth', type=int,  default=6, help='depth of the multimodal transformer')
    parser.add_argument('--dim_head', type=int, default=64, help='dimension per attention head')
    parser.add_argument('--heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--caption_loss_weight', type=float, default=1., help='weight on the autoregressive caption loss')
    parser.add_argument('--contrastive_loss_weight', type=float, default=1., help='weight on the contrastive loss between image and text CLS embeddings')
    parser.add_argument('--text_encoder_pretrained', type=bool, default=True, help='text encoder pretrained')
    parser.add_argument('--text_encoder', type=str, default='michiyasunaga/BioLinkBERT-base', help='text encoder')
    
    # optimizer
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    
    # training
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--save_path', type=str, default='/home/save/', help='save path')
    parser.add_argument('--save_name', type=str, default='model', help='save name')
    parser.add_argument('--log_interval', type=int, default=100, help='log interval')
    parser.add_argument('--eval_interval', type=int, default=100, help='eval interval')
    parser.add_argument('--save_interval', type=int, default=100, help='save interval')
    
    # device
    parser.add_argument('--device', type=str, default='cuda', help='device')
    
    # data
    parser.add_argument('--in_chans', type=int, default=12, help='in chans')
    parser.add_argument('--num_classes', type=int, default=5, help='num classes')
    
    # distributed
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank')
    parser.add_argument('--distributed', type=bool, default=False, help='distributed')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids')
    
    # wandb
    parser.add_argument('--wandb', type=bool, default=True, help='wandb')
    parser.add_argument('--wandb_project', type=str, default='project', help='wandb project')
    parser.add_argument('--wandb_name', type=str, default='name', help='wandb name')
    parser.add_argument('--wandb_group', type=str, default='group', help='wandb group')
    
    args = parser.parse_args()
    
    main(args)
    