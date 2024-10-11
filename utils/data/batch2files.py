import numpy as np
import json
from glob import glob
import pandas as pd
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description='Convert batched ECG files to single ECG files.')
parser.add_argument('--selected_modalities', type=str, default='', help='Path to the batched ECG files.')


def load_ecg_and_prompts(batch_number, output_dir='output', selected_modalities='all'):
    """
    Loads ECG signals and prompts from .npy and .json files respectively.

    Parameters:
    - batch_number: The identifier for the batch to load.
    - output_dir: Directory from where to load the files.

    Returns:
    A tuple containing:
    - A numpy array of ECG signals.
    - A list of dictionaries, where each dictionary contains prompt information.
    """
    if selected_modalities == '':
        suffix = ''
    else:
        suffix = f'_{selected_modalities}'
    
    # Construct filenames
    ecg_filename = os.path.join(output_dir, f'ecg_batch_{batch_number}.npy')
    prompts_filename = os.path.join(output_dir, f'prompts_batch_{batch_number}{suffix}.json')

    # Load ECG signals
    if selected_modalities == "":
        pass
        ecg_signals = np.load(ecg_filename, allow_pickle=True).astype(float)
    else:
        ecg_signals = np.array([1] * 5000)
    
    # Load prompts
    with open(prompts_filename, 'r') as f:
        prompts = json.load(f)
    
    return ecg_signals, prompts

def save_single_ecg(ecg_signal, save_index, output_dir='output'):
    np.save(os.path.join(output_dir, f'ecg_{save_index}.npy'), ecg_signal)

def main(args):
    if args.selected_modalities == '':
        suffix = 'all'
    else:
        suffix = args.selected_modalities
    
    ecg_file_path = '/scratch/bcht/hyu5/data/ecgs/batch/batch/'
    ecg_batch_files = glob(os.path.join(ecg_file_path, 'ecg_batch_*.npy'))[54:]
    batch_numbers = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in ecg_batch_files]
    print(ecg_batch_files) 
    save_index = 270000
    output_dir = '/scratch/bcht/hyu5/data/ecgs/files/'
    meta_output_dir = '/scratch/bcht/hyu5/data/ecgs/'
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    meta_data = {}
    
    for bn in batch_numbers:
        ecg_signals, prompts = load_ecg_and_prompts(bn, ecg_file_path, args.selected_modalities)
        print(ecg_signals.shape)
        print(len(prompts))
        
        for i, (ecg_signal, prompt) in tqdm(enumerate(zip(ecg_signals, prompts)), total=len(prompts)):
            if suffix == 'all':
                save_single_ecg(ecg_signal, save_index, output_dir)
            meta_data[save_index] = {"save_index": save_index, 'batch_number': bn, 'batch_index': i, 'prompt': prompt["prompt"]}
            save_index += 1
        print(f'Batch {bn} done.')
    meta_data_df = pd.DataFrame(meta_data).T
    meta_data_df.to_csv(os.path.join(meta_output_dir, 'meta_data_v2_{}.csv'.format(suffix)), index=False)
        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
