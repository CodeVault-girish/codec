from transformers import EncodecModel, AutoProcessor
import torchaudio
import torch

model = EncodecModel.from_pretrained("facebook/encodec_48khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")

wav_path = "# enter your wav file path"
audio_sample, sr = torchaudio.load(wav_path)

if sr != processor.sampling_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=processor.sampling_rate)
    audio_sample = resampler(audio_sample)

if audio_sample.shape[0] == 1:
    audio_sample = torch.cat([audio_sample, audio_sample], dim=0)

inputs = processor(raw_audio=audio_sample.squeeze().numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt")

encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])

#print(encoder_outputs)
print(encoder_outputs.audio_codes.shape)
#print(encoder_outputs.audio_codes.squeeze().float().mean(dim=0).shape)
#print(encoder_outputs.audio_codes.squeeze().float().mean(dim=0))
