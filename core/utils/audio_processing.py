import numpy as np
import matplotlib.pyplot as plt


def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav

def vis_waveform(wav):
    print("Shape of _waveform: {}".format(wav.size()))
    plt.figure()
    plt.plot(wav)
    plt.show()

def vis_spectrogram(specgram):
    print("Shape of spectrogram: {}".format(specgram.size()))
    plt.figure()
    plt.imshow(specgram.log2())
    plt.show()