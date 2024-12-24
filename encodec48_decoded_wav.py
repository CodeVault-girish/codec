from transformers import EncodecModel, AutoProcessor
import torchaudio
import torch

model = EncodecModel.from_pretrained("facebook/encodec_48khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")

wav_path = "01_CF56_1_0.wav"
audio_sample, sr = torchaudio.load(wav_path)
print("Original audio shape:", audio_sample.shape)

if sr != processor.sampling_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=processor.sampling_rate)
    audio_sample = resampler(audio_sample)

if audio_sample.shape[0] == 1:
    audio_sample = torch.cat([audio_sample, audio_sample], dim=0)  
print("Audio shape after ensuring stereo:", audio_sample.shape)

inputs = processor(raw_audio=audio_sample.numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt")

encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])

decoded_audio = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
print("Decoded audio shape:", decoded_audio.shape)

decoded_audio = decoded_audio.squeeze(0)  

if decoded_audio.ndim == 1:  
    decoded_audio = decoded_audio.unsqueeze(0)

decoded_audio = decoded_audio.to(torch.float32)

torchaudio.save("encodec48khz_decoded_wav.wav", decoded_audio.cpu(), processor.sampling_rate)
