import torch
import os
import glob
import argparse
import re
import numpy as np
import random
# from torchnlp.text_encoders import StaticTokenizerEncoder
from torch.autograd import Variable
# pip install pytorch-nlp==0.3.0
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from Utils.CustomMFCC import *


class DataUtils:
    def __init__(self):
        return

    @staticmethod
    def build_gid_lab_dict_libri(file_directory, gender_txt_file, folder_name):

        wav_file_list = glob.glob(file_directory, recursive=True)
        with open(gender_txt_file) as f:
            lines = f.readlines()
        dict = {}
        for i in range(len(lines)):
            if i == 0:
                continue
            line = lines[i]
            splited_line = line.split(',')
            dict[splited_line[0]] = splited_line[1].strip()
        lab_dict = {}
        new_wav_file_list = []
        for file_name in wav_file_list:
            indices = [_.start() for _ in re.finditer(folder_name, file_name)]
            speaker_id = file_name[indices[0] + len(folder_name) + 1: -4]
            new_wav_file_list.append(speaker_id + '.wav')
            lbl = 0
            if dict[speaker_id] == 'F':
                lbl = 1
            lab_dict[speaker_id + '.wav'] = lbl
        return new_wav_file_list, lab_dict

    @staticmethod
    def keep_gender(wav_file_list, gender):
        new_wav_file_list = []
        for file_name in wav_file_list:
            indices = [_.start() for _ in re.finditer('/', file_name)]
            speaker_id = file_name[indices[-2] + 1: indices[-1]]
            if speaker_id[0] == gender:
                new_wav_file_list.append(file_name)
        return new_wav_file_list

    @staticmethod
    def get_sid_class_labels(wav_file_list):
        """
        extracts TIMIT labels for list of wav file names given to it for SID task
        :param wav_file_list:
        :return:
        """
        class_lbls = set()
        for file_name in wav_file_list:
            indices = [_.start() for _ in re.finditer('/', file_name)]
            speaker_id = file_name[indices[-2]+1: indices[-1]]
            class_lbls.add(speaker_id)
        return list(class_lbls)

    @staticmethod
    def distinct_gid_train_val(wav_lst, tr_spk_lbls):
        wav_lst_tr = []
        wav_lst_val = []
        random.shuffle(tr_spk_lbls)
        for lbl_ind, lbl in enumerate(tr_spk_lbls):
            for file_name in wav_lst:
                if lbl in file_name:
                    if lbl_ind < int(0.2 * tr_spk_lbls.__len__()):
                        wav_lst_val.append(file_name)
                    else:
                        wav_lst_tr.append(file_name)
        return wav_lst_tr, wav_lst_val

    @staticmethod
    def build_gid_lab_dict(wav_lst):
        """
        builds a dictionary which determines the gender label of each file name for GID task
        :param wav_lst:
        :param class_lbls:
        :return:
        """
        lab_dict = {}
        for file_name in wav_lst:
            indices = [_.start() for _ in re.finditer('/', file_name)]
            speaker_id = file_name[indices[-2] + 1: indices[-1]]
            lbl = 0
            if speaker_id[0] == 'F':
                lbl = 1
            lab_dict[file_name] = lbl
        return lab_dict

    @staticmethod
    def build_gid_lab_dict_voxceleb(data_folder_path):
        lab_dict = {}
        wav_lst = []
        for path, subdirs, files in os.walk(data_folder_path):
            for name in files:
                full_path = os.path.join(path, name)
                label = full_path[len(data_folder_path) + 8]
                rest_path = full_path[len(data_folder_path):]
                wav_lst.append(rest_path)
                lab_dict[rest_path] = 0
                if label == 'F':
                    lab_dict[rest_path] = 1
                # print(full_path + '  ' + label + ' ' + rest_path)
        return lab_dict, wav_lst

    @staticmethod
    def build_gid_lab_dict_ravdess(data_folder_path):
        lab_dict = {}
        wav_lst = []
        for path, subdirs, files in os.walk(data_folder_path + 'Audio_Speech_Actors_01-24/'):
            for name in files:
                full_path = os.path.join(path, name)
                lab_dict[full_path] = 0
                if '_F' in full_path:
                    lab_dict[full_path] = 1
                wav_lst.append(full_path)
        return lab_dict, wav_lst

    @staticmethod
    def get_core_wav_test_files(wav_file_names):
        """
        returns core test wave file names
        :param wav_file_names:
        :return:
        """
        test_speakers = [
            'MDAB0', 'MWBT0', 'FELC0', 'MTAS1', 'MWEW0', 'FPAS0', 'MJMP0', 'MLNT0', 'FPKT0', 'MLLL0', 'MTLS0', 'FJLM0',
            'MBPM0', 'MKLT0', 'FNLP0', 'MCMJ0', 'MJDH0', 'FMGD0', 'MGRT0', 'MNJM0', 'FDHC0', 'MJLN0', 'MPAM0', 'FMLD0']
        core_wav_file_names = []
        for file_name in wav_file_names:
            for test_speaker in test_speakers:
                if test_speaker in file_name and 'SA1' not in file_name and 'SA2' not in file_name:
                    core_wav_file_names.append(file_name)
                    break
        return core_wav_file_names

    @staticmethod
    def remove_sa_wav_files(wav_file_names):
        """
        removes SA files from data
        :param wav_file_names:
        :return:
        """
        new_wav_file_names = []
        for file_name in wav_file_names:
            if 'SA1' not in file_name and 'SA2' not in file_name:
                new_wav_file_names.append(file_name.replace('\\', '/'))
        return new_wav_file_names

    @staticmethod
    def read_wav_file_names(data_path, train=1):
        """
        reads wav file names in the path specified, by train = 1, train data will be readed and with train != 1, test data
        :param data_path:
        :param train:
        :return:
        """
        if train == 1:
            wav_file_names = glob.glob(os.path.join(data_path, "TRAIN/**/*.WAV"), recursive=True)
        else:
            wav_file_names = glob.glob(os.path.join(data_path, "TEST/**/*.WAV"), recursive=True)
        return wav_file_names

    @staticmethod
    def get_mfcc_features(wav_lst, lab_dict, reduce_frames=True, filters_loc=None,
                          filters_f1_f2=None, filters_abs=None):
        X_c = []
        y_c = []
        X_f = []
        y_f = []
        cus_mfcc = CustomMFCC()
        freq_min = 0
        sample_rate = 16000
        mel_filter_num = 80
        filters_loc = filters_loc
        filters_f1_f2 = filters_f1_f2
        filters_abs = filters_abs
        FFT_size = 3200
        freq_high = sample_rate / 2

        if filters_loc is None:
            filter_points, mel_freqs = cus_mfcc.get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size,
                                                              sample_rate=sample_rate)
            filters = cus_mfcc.get_filters(filter_points, FFT_size)
            # taken from the librosa library
            enorm = 2.0 / (mel_freqs[2:mel_filter_num + 2] - mel_freqs[:mel_filter_num])
            filters *= enorm[:, np.newaxis]
        else:
            filters, freqs, enorm = cus_mfcc.get_filters_custom(filters_f1_f2, filters_abs, FFT_size)
            # filters *= enorm[:, np.newaxis]

        # plt.figure(figsize=(15, 4))
        # for n in range(filters.shape[0]):
        #     plt.plot(filters[n])
        # plt.show()

        for wav_path in wav_lst:
            # (rate, sig) = wav.read(wav_path)
            # mfcc_feat = mfcc(sig, rate)  # , winlen=0.2, winstep=0.01, nfft=3200, numcep=40, nfilt=80)
            mfcc_feat = cus_mfcc.do(wav_path, filters, hop_size=10, FFT_size=FFT_size)
            for i in range(mfcc_feat.shape[0]):
                X_f.append(mfcc_feat[i])
                y_f.append(lab_dict[wav_path])
            features = np.sum(mfcc_feat, axis=0) / mfcc_feat.shape[0]
            X_c.append(features)
            y_c.append(lab_dict[wav_path])
        if reduce_frames:
            n_selected_frames = 400
            sel_indices = [random.randint(0, y_f.__len__()) for i in range(n_selected_frames)]
            X_f = [X_f[i] for i in sel_indices]
            y_f = [y_f[i] for i in sel_indices]
        return X_c, y_c, X_f, y_f

    @staticmethod
    def set_lbl_str(lbls, str_lbls):
        new_lbls = []
        for lbl in lbls:
            new_lbls.append(str_lbls[lbl])
        return new_lbls



