from transformers import EncodecModel, AutoProcessor
import torchaudio
import torch

model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

wav_path = "# call_275_0.wav"
audio_sample, sr = torchaudio.load(wav_path)
print(audio_sample.shape)

if audio_sample.shape[0] > 1:  
    audio_sample = torch.mean(audio_sample, dim=0, keepdim=True)

if sr != processor.sampling_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=processor.sampling_rate)
    audio_sample = resampler(audio_sample)

inputs = processor(raw_audio=audio_sample.squeeze().numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt")

encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
print(encoder_outputs.audio_codes.shape)

decoded_audio = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
print(decoded_audio.shape)

decoded_audio = decoded_audio.squeeze(0) 

if decoded_audio.ndim == 1: 
    decoded_audio = decoded_audio.unsqueeze(0)

decoded_audio = decoded_audio.to(torch.float32)

torchaudio.save("decoded_encodec24khz_audio.wav", decoded_audio.cpu(), processor.sampling_rate)




# from transformers import EncodecModel, AutoProcessor
# import torchaudio
# import torch

# model = EncodecModel.from_pretrained("facebook/encodec_24khz")
# processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# wav_path = "/home/girish/Girish/RESEARCH/911_data/911_first6sec/call_275_0.wav"
# audio_sample, sr = torchaudio.load(wav_path)
# print(audio_sample.shape)

# if sr != processor.sampling_rate:
#     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=processor.sampling_rate)
#     audio_sample = resampler(audio_sample)

# inputs = processor(raw_audio=audio_sample.squeeze().numpy(), sampling_rate=processor.sampling_rate, return_tensors="pt")

# encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
# print(encoder_outputs.audio_codes.shape)
# decoded_audio = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
# print(decoded_audio.shape)

# decoded_audio = decoded_audio.squeeze(0)  

# if decoded_audio.ndim == 1:  
#     decoded_audio = decoded_audio.unsqueeze(0)

# decoded_audio = decoded_audio.to(torch.float32)

# torchaudio.save("decoded_audio.wav", decoded_audio.cpu(), processor.sampling_rate)
