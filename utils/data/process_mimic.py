import pandas as pd
import wfdb
import os
import numpy as np
import json
from tqdm import tqdm

# Load waveform dictionary
with open("../../rag/mimic_waveform_dict.json", "r") as f:
    waveform_dict = json.load(f)

def compile_ecg_prompt(gender, anchor_age, ecg_measurements):
    prompt = f"This patient is a {anchor_age} year-old {gender}. "
    prompt += "ECG Findings: "
    prompt += ", ".join(ecg_measurements) + "."
    waveform_prompt = "\n".join([waveform_dict[measure] for measure in ecg_measurements if measure])
    prompt += "\nPotential ECG waveforms: \n" + waveform_prompt
    return prompt

def save_ecg_and_prompts(batch_number, ecg_signals, prompts, output_dir=''):
    os.makedirs(output_dir, exist_ok=True)
    
    ecg_filename = os.path.join(output_dir, f'ecg_batch_{batch_number}.npy')
    np.save(ecg_filename, np.array(ecg_signals))
    
    prompts_filename = os.path.join(output_dir, f'prompts_batch_{batch_number}_waveforms.json')
    with open(prompts_filename, 'w') as f:
        json.dump(prompts, f, indent=4)
    
    print(f'Batch {batch_number} saved. ECGs: {ecg_filename}, Prompts: {prompts_filename}')

def load_ecg_and_prompts(batch_number, output_dir='output'):
    ecg_filename = os.path.join(output_dir, f'ecg_batch_{batch_number}.npy')
    prompts_filename = os.path.join(output_dir, f'prompts_batch_{batch_number}.json')

    ecg_signals = np.load(ecg_filename)
    
    with open(prompts_filename, 'r') as f:
        prompts = json.load(f)
    
    return ecg_signals, prompts

def check_cardiac_notes(notes):
    cardic_keywords = ["CV:", "CV- ", "HEART:", "Heart:", "Heart -", "Cardiac:", "CARDIAC:", "COR:", "heart ", "CARDIOVASCULAR:", "CV -", "Cards:", "Cardiovascular:", "Cardiovascular ", "Cardiovascular:", "Cardiovasc:"]
    return any(keyword in notes for keyword in cardic_keywords)

def main():
    # Load data
    data_path = '/path/to/physionet.org/files/1.0/'
    ecg_meta = pd.read_csv('/path/to/ecg_meta.csv')
    ecg_machine_measure = pd.read_csv('/path/to/machine_measurements.csv')
    subjects_info = pd.read_csv('/path/to/subjects_info.csv')

    gender_dict = {'M': "Male", 'F': "Female"}

    batch_size = 5000
    batch_number = 0

    current_batch_signal = []
    current_batch_prompt = []

    for i, row in tqdm(ecg_meta.iterrows(), total=len(ecg_meta)):
        try:
            subject_id = row['subject_id']
            study_id = row['study_id']
            waveform_path = row['waveform_path']
            record = wfdb.rdrecord(os.path.join(data_path, waveform_path))
            record_signal = record.p_signal
            
            measure_data = ecg_machine_measure[(ecg_machine_measure['study_id'] == study_id) & (ecg_machine_measure['subject_id'] == subject_id)]
            ecg_measure_data = measure_data.iloc[0, 4:22]
            ecg_measure_data = ecg_measure_data[~ecg_measure_data.isnull()].values
            
            demographic_data = subjects_info[subjects_info['subject_id'] == subject_id]
            gender = gender_dict[demographic_data.gender.values[0]]
            anchor_age = demographic_data.anchor_age.values[0]
            
            text_prompt = compile_ecg_prompt(gender, anchor_age, ecg_measure_data)
            
            current_batch_signal.append(record_signal)
            current_batch_prompt.append({"prompt": text_prompt, "meta": {"subject_id": subject_id, "study_id": study_id}})
            
            if len(current_batch_signal) == batch_size:
                save_ecg_and_prompts(batch_number, current_batch_signal, current_batch_prompt)
                batch_number += 1
                current_batch_signal = []
                current_batch_prompt = []
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue

    # Save any remaining data in the last batch
    if current_batch_signal:
        save_ecg_and_prompts(batch_number, current_batch_signal, current_batch_prompt)

    # Process ECG notes
    ecg_notes = pd.read_csv('path/to/discharge.csv.gz')
    ecg_notes_available = ecg_notes[ecg_notes.text.apply(check_cardiac_notes)]
    print(f"Number of available ECG notes: {len(ecg_notes_available)}")

if __name__ == "__main__":
    main()