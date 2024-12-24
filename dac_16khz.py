import os
import pandas as pd
import torchaudio
import torch
from transformers import DacModel, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DacModel.from_pretrained("descript/dac_16khz").to(device)
processor = AutoProcessor.from_pretrained("descript/dac_16khz")
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
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU

        encoder_outputs = model.encode(inputs["input_values"])

        features = encoder_outputs.audio_codes.squeeze().float().mean(dim=0).cpu().numpy()

        data.append([filename] + features.tolist())

df = pd.DataFrame(data, columns=["filename"] + [f"feature_{i}" for i in range(len(features))])

csv_output_path = "# enter your output file path.csv"
df.to_csv(csv_output_path, index=False)

print(f"Features saved to {csv_output_path}")






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