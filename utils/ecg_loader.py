import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer



class Jitter(object):
    def __init__(self, sigma=0.03):
        self.sigma = sigma

    def jitter(self, x, sigma):
        # Jitter is added to every point in the time series data, so no change is needed based on channel_first
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    
    def __call__(self, x):
        return self.jitter(x, self.sigma)


class Scaling(object):
    def __init__(self, sigma=0.1, channel_first=False):
        self.sigma = sigma
        self.channel_first = channel_first

    def scaling(self, x, sigma):
        if self.channel_first:
            # For (B, C, L), generate a factor for each channel and broadcast across L
            factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], 1))
        else:
            # For (B, L, C), generate a factor for each channel and broadcast across L
            factor = np.random.normal(loc=1., scale=sigma, size=(1, x.shape[1]))
        
        return np.multiply(x, factor)
    
    def __call__(self, x):
        return self.scaling(x, self.sigma)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample = torch.Tensor(sample)
        return sample

class ECGDataset(Dataset):
    def __init__(self, meta_path, file_path, augmentation=True):
        self.ecg_meta = pd.read_csv(meta_path)
        self.file_path = file_path
        if augmentation == True:
            self.transform = transforms.Compose([
                Jitter(0.003),
                Scaling(0.003),
                ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                ToTensor()
            ])
        
    def __len__(self):
        return len(self.ecg_meta)

    def __getitem__(self, idx):
        ecg_info = self.ecg_meta.iloc[idx]
        save_index = ecg_info['save_index']
        prompt = ecg_info['prompt']
        ecg_path = f'{self.file_path}/ecg_{save_index}.npy'
        ecg_signal = np.load(ecg_path, allow_pickle=True).astype(float)
        
        # Channel-wise imputation for NaNs
        if np.isnan(ecg_signal).any():
            for i in range(ecg_signal.shape[-1]): # Loop through each channel
                if np.isnan(ecg_signal[:, i]).any(): # Check if the current channel has NaNs
                    mean_val = np.nanmean(ecg_signal[:, i])
                    ecg_signal[:, i] = np.nan_to_num(ecg_signal[:, i], nan=mean_val)
        
        ecg_signal = self.transform(ecg_signal)
        
        # Standard normalization process
        mean = ecg_signal.mean(dim=[0, 1], keepdim=True)
        std = ecg_signal.std(dim=[0, 1], keepdim=True)
        ecg_signal = (ecg_signal - mean) / (std + 1e-6)
        
        return ecg_signal, prompt

def tokenize_collate_fn(batch, tokenizer):
    ecg_signals, prompts = zip(*batch)
    ecg_signals = torch.stack(ecg_signals)
    prompts_tokenized = tokenizer(prompts, padding=True, truncation=True, max_length=500, return_tensors='pt')
    return ecg_signals, prompts_tokenized


def get_ecg_loader(meta_path, file_path, batch_size, collate_fn=None, num_workders=4):
    dataset = ECGDataset(meta_path, file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workders, pin_memory=True, collate_fn=collate_fn)



# def wrapper_collate_fn(batch):
#     return tokenize_collate_fn(batch, tokenizer)
# tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base', model_max_length=512)
# loader = get_ecg_loader('/rdf/data/physio_clip/ecg/meta_data_batch1.csv', '/rdf/data/physio_clip/ecg/files', 32, wrapper_collate_fn)
# for batch in tqdm(loader):
#     print(batch[1]['attention_mask'].shape)
#     print(batch[0].shape, batch[1]['input_ids'], batch[1]['attention_mask'].shape, batch[1]['token_type_ids'].shape)
#     pass
