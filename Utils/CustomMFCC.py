import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import matplotlib.pyplot as plt


class CustomMFCC:
    """
    src: https://www.kaggle.com/code/ilyamich/mfcc-implementation-and-tutorial
    """
    def __init__(self):
        return

    def do(self, path, filters, hop_size=15, FFT_size=2048):
        # hop_size in ms

        sample_rate, audio = wavfile.read(path)
        audio = self.normalize_audio(audio)

        # Audio Framing
        audio_framed = self.frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)

        # Convert to frequency domain
        window = get_window("hann", FFT_size, fftbins=True)
        audio_win = audio_framed * window
        audio_winT = np.transpose(audio_win)
        audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
        for n in range(audio_fft.shape[1]):
            audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
        audio_fft = np.transpose(audio_fft)

        # Calculate signal power
        audio_power = np.square(np.abs(audio_fft))

        # Filter the signal
        audio_filtered = np.dot(filters, np.transpose(audio_power))
        audio_log = 10.0 * np.log10(audio_filtered + 0.0000001)

        # Generate the Cepstral Coefficents
        dct_filter_num = 40
        mel_filter_num = filters.shape[0]
        dct_filters = self.dct(dct_filter_num, mel_filter_num)

        cepstral_coefficents = np.dot(dct_filters, audio_log)
        return np.transpose(cepstral_coefficents)

    def normalize_audio(self, audio):
        audio = audio / np.max(np.abs(audio))
        return audio

    def frame_audio(self, audio, FFT_size=2048, hop_size=10, sample_rate=16000):
        # hop_size in ms
        audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
        frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
        frame_num = int((len(audio) - FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num, FFT_size))
        for n in range(frame_num):
            frames[n] = audio[n * frame_len:n * frame_len + FFT_size]

        return frames

    def freq_to_mel(self, freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def met_to_freq(self, mels):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    def get_filter_points(self, fmin, fmax, mel_filter_num, FFT_size, sample_rate=16000):
        fmin_mel = self.freq_to_mel(fmin)
        fmax_mel = self.freq_to_mel(fmax)
        mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num + 2)
        freqs = self.met_to_freq(mels)
        return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

    def get_filter_points_custom(self, FFT_size, filters_loc, sample_rate=16000):
        freqs = (sample_rate * np.angle(filters_loc)) / (2 * np.pi)
        freqs.sort()
        new_freqs = []
        new_freqs.append(0)
        for i in range(freqs.__len__() - 1):
            new_freqs.append((freqs[i] + freqs[i+1]) / 2)
        new_freqs.append(sample_rate/2)
        new_freqs = np.asarray(new_freqs)
        return np.floor((FFT_size + 1) / sample_rate * new_freqs).astype(int), new_freqs

    def get_filters(self, filter_points, FFT_size):
        filters = np.zeros((len(filter_points) - 2, int(FFT_size // 2 + 1)))
        for n in range(len(filter_points) - 2):
            filters[n, filter_points[n]: filter_points[n + 1]] = np.linspace(0, 1,
                                                                             filter_points[n + 1] - filter_points[n])
            filters[n, filter_points[n + 1]: filter_points[n + 2]] = np.linspace(1, 0,
                                                                                 filter_points[n + 2] - filter_points[
                                                                                     n + 1])
        return filters

    def get_filters_custom(self, filters_f1_f2, filters_abs, FFT_size):
        filters = np.zeros((len(filters_f1_f2), int(FFT_size // 2 + 1)))
        freqs = []
        remove_indices = set()
        for n in range(len(filters_f1_f2)):
            f1_point = int(FFT_size * filters_f1_f2[n][0] / 16000)
            if f1_point % 2 == 1:
                f1_point += 1
            f2_point = int(FFT_size * filters_f1_f2[n][1] / 16000)
            if f2_point % 2 == 1:
                f2_point += 1

            fc_point = int((f1_point + f2_point) / 2)

            freqs.append((filters_f1_f2[n][0] + filters_f1_f2[n][1]) / 2)
            # print(str(f1_point) + ', ' + str(f2_point) + ', ' + str(fc_point))
            start_point = 2*f1_point - fc_point
            end_point = fc_point
            start_val = 0
            end_val = filters_abs[n]
            if start_point < 0:
                start_point = 0
                start_val = (end_val/2*fc_point - end_val*f1_point) / (fc_point-f1_point)
            num = end_point - start_point
            if num > 0:
                filters[n, start_point: end_point] = np.linspace(start_val, end_val, num)
            else:
                remove_indices.add(n)

            start_point = fc_point
            end_point = 2*f2_point - fc_point
            start_val = filters_abs[n]
            end_val = 0
            if end_point >= FFT_size//2:
                end_point = FFT_size//2
                if f2_point != fc_point:
                    end_val = (start_val/2 * (FFT_size//2 - fc_point)) / (fc_point - f2_point) + start_val
            num = end_point - start_point
            if num > 0:
                filters[n, start_point: end_point] = np.linspace(start_val, end_val, num)
            else:
                remove_indices.add(n)
            # num = int((f1_point + f2_point) / 2) - f1_point
            # filters[n, f1_point: f1_point+num] = np.linspace(0, filters_abs[n], num)
            # filters[n, f1_point+num: f2_point] = np.linspace(filters_abs[n], 0, num)

        filters = np.asarray([j for i, j in enumerate(filters) if i not in list(remove_indices)])
        freqs = np.asarray([j for i, j in enumerate(freqs) if i not in list(remove_indices)])
        filters_f1_f2 = np.asarray([j for i, j in enumerate(filters_f1_f2) if i not in list(remove_indices)])
        sorted_indices = np.argsort(freqs)
        freqs = freqs[sorted_indices]
        filters = filters[sorted_indices]
        filters_f1_f2 = filters_f1_f2[sorted_indices]
        enorm = 2 / (np.asarray(filters_f1_f2)[:, 1] - np.asarray(filters_f1_f2)[:, 0])
        return filters, freqs, enorm



    def dct(self, dct_filter_num, filter_len):
        basis = np.empty((dct_filter_num, filter_len))
        basis[0, :] = 1.0 / np.sqrt(filter_len)
        samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)
        for i in range(1, dct_filter_num):
            basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        return basis
