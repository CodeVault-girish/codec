from transformers import DacModel, AutoProcessor
import torchaudio
import torch
import torch.nn.functional as F

model = DacModel.from_pretrained("descript/dac_16khz")
processor = AutoProcessor.from_pretrained("descript/dac_16khz")

wav_path = "# enter your wav file path"
audio_sample, sr = torchaudio.load(wav_path)

if sr != processor.sampling_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=processor.sampling_rate)
    audio_sample = resampler(audio_sample)

inputs = processor(raw_audio=audio_sample.squeeze().numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt")

encoder_outputs = model.encode(inputs["input_values"])

audio_codes = encoder_outputs.audio_codes
print(f"Original shape of audio codes: {audio_codes.shape}")

audio_codes = F.interpolate(audio_codes.permute(0, 2, 1), size=1024, mode='linear').permute(0, 2, 1)
print(f"Interpolated shape of audio codes: {audio_codes.shape}")

audio_codes = audio_codes.float()

decoded_audio = model.decode(audio_codes)[0]

if decoded_audio.ndim == 1:  # If it's 1D, assume mono and add a channel dimension
    decoded_audio = decoded_audio.unsqueeze(0)

decoded_audio = decoded_audio.to(torch.float32)

torchaudio.save("decoded_dac_audio.wav", decoded_audio.cpu(), processor.sampling_rate)
