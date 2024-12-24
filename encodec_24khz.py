# ---------------------- encodec 24khz on cuda for whole folder -----------------------

import os
import pandas as pd
import torchaudio
import torch
from transformers import EncodecModel, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

wav_folder = "# enter your folder path"

data = []

for filename in os.listdir(wav_folder):
    if filename.endswith(".wav"):
        file_path = os.path.join(wav_folder, filename)
        
        audio_sample, sr = torchaudio.load(file_path)
        
        if sr != processor.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=processor.sampling_rate)
            audio_sample = resampler(audio_sample)


        inputs = processor(raw_audio=audio_sample.squeeze().numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}  

        encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
        
        features = encoder_outputs.audio_codes.squeeze().float().mean(dim=0).cpu().numpy()
        
        data.append([filename] + features.tolist())

df = pd.DataFrame(data, columns=["filename"] + [f"feature_{i}" for i in range(len(features))])
csv_output_path = "# enter your output file path .csv"
df.to_csv(csv_output_path, index=False)

print(f"Features saved to {csv_output_path}")







# ---------------------- encodec 24khz on cuda -----------------------


# from transformers import EncodecModel, AutoProcessor
# import torchaudio
# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
# processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# wav_path = "/home/girish/Girish/RESEARCH/depression_DATA_wavCODEC/android_segmented_5s/reading_hc/49_CM54_4_0.wav"
# audio_sample, sr = torchaudio.load(wav_path)

# if sr != processor.sampling_rate:
#     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=processor.sampling_rate)
#     audio_sample = resampler(audio_sample)

# inputs = processor(raw_audio=audio_sample.squeeze().numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt")
# inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU

# encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])

# print(encoder_outputs)
# print(encoder_outputs.audio_codes.squeeze().float().mean(dim=0).shape)
# print(encoder_outputs.audio_codes.squeeze().float().mean(dim=0))




# ---------------------- encodec 24khz -----------------------


# from transformers import EncodecModel, AutoProcessor
# import torchaudio
# import torch

# model = EncodecModel.from_pretrained("facebook/encodec_24khz")
# processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# wav_path = "/home/girish/Girish/RESEARCH/depression_DATA_wavCODEC/android_segmented_5s/reading_hc/49_CM54_4_0.wav"
# audio_sample, sr = torchaudio.load(wav_path)

# if sr != processor.sampling_rate:
#     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=processor.sampling_rate)
#     audio_sample = resampler(audio_sample)

# inputs = processor(raw_audio=audio_sample.squeeze().numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt")

# encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])

# print(encoder_outputs)
# print(encoder_outputs.audio_codes.squeeze().float().mean(dim=0).shape)
# print(encoder_outputs.audio_codes.squeeze().float().mean(dim=0))