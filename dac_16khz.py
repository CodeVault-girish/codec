import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import DacModel, AutoProcessor
from torchaudio.transforms import Resample

folder_path = "wav folder path"
output_file = "Dac.csv"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DacModel.from_pretrained("descript/dac_16khz").to(device)
processor = AutoProcessor.from_pretrained("descript/dac_16khz")

def preprocess_audio(audio_path, target_sample_rate):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        return waveform, target_sample_rate
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None, None

def extract_features(audio_path, model, processor, device):
    target_sample_rate = processor.sampling_rate
    waveform, _ = preprocess_audio(audio_path, target_sample_rate)
    if waveform is None:
        return None

    try:
        inputs = processor(raw_audio=waveform.squeeze().numpy(), sampling_rate=target_sample_rate, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            encoder_outputs = model.encode(inputs["input_values"])
        
        audio_codes = encoder_outputs.audio_codes.squeeze().float().mean(dim=0).cpu().numpy()
        return audio_codes
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

all_features = []
filenames = []

for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        
        features = extract_features(file_path, model, processor, device)
        if features is not None:
            
            all_features.append(features)
            filenames.append(filename)

features_df = pd.DataFrame(all_features)
features_df.insert(0, 'filename', filenames)  

features_df.to_csv(output_file, index=False)
print(f"Saved all features to {output_file}")
















# import os
# import pandas as pd
# import torchaudio
# import torch
# from transformers import DacModel, AutoProcessor

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = DacModel.from_pretrained("descript/dac_16khz").to(device)
# processor = AutoProcessor.from_pretrained("descript/dac_16khz")
# wav_folder = "# enter your folder path"

# data = []
# for filename in os.listdir(wav_folder):
#     if filename.endswith(".wav"):
#         file_path = os.path.join(wav_folder, filename)
        
#         audio_sample, sr = torchaudio.load(file_path)
        
#         if sr != processor.sampling_rate:
#             resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=processor.sampling_rate)
#             audio_sample = resampler(audio_sample)

#         inputs = processor(raw_audio=audio_sample.squeeze().numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt")
#         inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU

#         encoder_outputs = model.encode(inputs["input_values"])

#         features = encoder_outputs.audio_codes.squeeze().float().mean(dim=0).cpu().numpy()

#         data.append([filename] + features.tolist())

# df = pd.DataFrame(data, columns=["filename"] + [f"feature_{i}" for i in range(len(features))])

# csv_output_path = "# enter your output file path.csv"
# df.to_csv(csv_output_path, index=False)

# print(f"Features saved to {csv_output_path}")






# ---------------------- dac 16khz -----------------------

# from datasets import load_dataset, Audio
# from transformers import DacModel, AutoProcessor
# import pandas as pd
# import torchaudio
# import torch
# model = DacModel.from_pretrained("descript/dac_16khz")
# processor = AutoProcessor.from_pretrained("descript/dac_16khz")
# wav_path = "/home/girish/Girish/RESEARCH/depression_DATA_wavCODEC/android_segmented_5s/reading_hc/49_CM54_4_0.wav"
# audio_sample, sr = torchaudio.load(wav_path)

# if sr != processor.sampling_rate:
#     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=processor.sampling_rate)
#     audio_sample = resampler(audio_sample)

# #audio_sample = librispeech_dummy[-1]["audio"]["array"]
# #inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")
# inputs = processor(raw_audio=audio_sample.squeeze().numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt")
# encoder_outputs = model.encode(inputs["input_values"])
# audio_codes = encoder_outputs.audio_codes
# #print(audio_codes.shape)
# #print(audio_codes)
# print(encoder_outputs)
# print(encoder_outputs.audio_codes.squeeze().float().mean(dim=0).shape)
# #print(encoder_outputs.audio_codes.squeeze().float().mean(dim=0))