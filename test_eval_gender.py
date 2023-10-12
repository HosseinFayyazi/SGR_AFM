import pickle
import torch
import soundfile as sf
import argparse
import io
from Utils.GeneralUtils import *
from Utils.DataUtils import *
import librosa
import random


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='IO/final/G_cnn/1_saved_model.pth_best.pth')
parser.add_argument('--cfg_file', type=str, default='IO/final/G_cnn/KernelCNN_TIMIT.cfg')
parser.add_argument('--dataset', type=str, default='RAVDESS')

args = parser.parse_args()


models = [
    'IO/final/G_cnn_lib/1_saved_model.pth_best.pth',
    'IO/final/G_sinc_lib/1_saved_model.pth_best.pth',
    'IO/final/G_sinc2_lib/1_saved_model.pth_best.pth',
    'IO/final/G_gamma_lib/1_saved_model.pth_best.pth',
    'IO/final/G_gauss_lib/1_saved_model.pth_best.pth',
    'IO/final/G_gauss_cascade_lib/1_saved_model.pth_best.pth',
    'IO/final/G_mel_lib/1_saved_model.pth_best.pth',
    'IO/final/G_custom_freeze_lib/1_saved_model.pth_best.pth'
]
cfgs = [
    'IO/final/G_cnn_lib/KernelCNN_LIB.cfg',
    'IO/final/G_sinc_lib/KernelSinc_LIB.cfg',
    'IO/final/G_sinc2_lib/KernelSinc2_LIB.cfg',
    'IO/final/G_gamma_lib/KernelGamma_LIB.cfg',
    'IO/final/G_gauss_lib/KernelGauss_LIB.cfg',
    'IO/final/G_gauss_cascade_lib/KernelGaussCascade_LIB.cfg',
    'IO/final/G_mel_lib/Mel_LIB.cfg',
    'IO/final/G_custom_freeze_lib/KernelCustomFreeze_LIB.cfg'
]

for ii, model_path in enumerate(models):
    options = GeneralUtils.read_conf(cfgs[ii])
    with open(model_path, 'rb') as file:
        trainer = pickle.load(file)

    # with open(args.save_path, 'rb') as file:
    #     trainer = CPU_Unpickler(file).load()
    # trainer.gpu = False

    if args.dataset == 'voxceleb':
        data_folder_path = '../../Data/voxceleb/wav/'
        lab_dict, wav_lst_te = DataUtils.build_gid_lab_dict_voxceleb(data_folder_path)
        trainer.lab_dict = lab_dict
        trainer.data_folder = data_folder_path
        snt_te = len(wav_lst_te)
    elif args.dataset == 'librispeech':
        data_folder_path = '../../Data/LibriSpeech/test1/'
        gender_txt_file = '../../Data/LibriSpeech/GENDERS.txt'
        file_directory = '../../Data/LibriSpeech/test1/*.wav'
        folder_name = 'test1'
        wav_lst_te, lab_dict = DataUtils.build_gid_lab_dict_libri(file_directory, gender_txt_file, folder_name)
        trainer.lab_dict = lab_dict
        trainer.data_folder = data_folder_path
        snt_te = len(wav_lst_te)
    elif args.dataset == 'TIMIT':
        data_folder_path = '../../Data/TIMIT_NORM'
        wav_lst_tr = DataUtils.read_wav_file_names(data_folder_path, train=1)
        wav_lst_tr = DataUtils.remove_sa_wav_files(wav_lst_tr)
        wav_lst_te = DataUtils.read_wav_file_names(data_folder_path, train=0)
        wav_lst_te_comp = DataUtils.remove_sa_wav_files(wav_lst_te)
        wav_lst_te_core = GeneralUtils.get_core_wav_test_files(wav_lst_te_comp)
        wav_lst_te = wav_lst_te_comp
        lab_dict = DataUtils.build_gid_lab_dict(wav_lst_tr + wav_lst_te)
        trainer.lab_dict = lab_dict
        trainer.data_folder = data_folder_path
        # wav_lst = wav_lst_tr + wav_lst_te
        # if args.gender == 'M' or args.gender == 'F':
        #     wav_lst = DataUtils.keep_gender(wav_lst, args.gender)
        # #train samples = 3692, #test samples = 1344
        lbls = ['M', 'F']
        tr_spk_lbls = DataUtils.get_sid_class_labels(wav_lst_tr)
        wav_lst_tr, wav_lst_val = DataUtils.distinct_gid_train_val(wav_lst_tr, tr_spk_lbls)

        # #train samples = 2960|370, #val_samples = 736|92, #test samples = 1344|168
        snt_tr = len(wav_lst_tr)
        snt_val = len(wav_lst_val)
        snt_te = len(wav_lst_te)

    elif args.dataset == 'SUB_TIMIT':
        num_rnd = 8
        data_folder_path = '../../Data/TIMIT_NORM'
        wav_lst_tr = DataUtils.read_wav_file_names(data_folder_path, train=1)
        wav_lst_tr = DataUtils.remove_sa_wav_files(wav_lst_tr)
        wav_lst_te = DataUtils.read_wav_file_names(data_folder_path, train=0)
        wav_lst_te_comp = DataUtils.remove_sa_wav_files(wav_lst_te)
        wav_lst_te_core = GeneralUtils.get_core_wav_test_files(wav_lst_te_comp)
        random.shuffle(wav_lst_te_comp)
        wav_lst_F = DataUtils.keep_gender(wav_lst_te_comp, 'F')
        wav_lst_M = DataUtils.keep_gender(wav_lst_te_comp, 'M')
        wav_lst_te = wav_lst_F[0: num_rnd] + wav_lst_M[0: num_rnd]
        random.shuffle(wav_lst_te)
        lab_dict = DataUtils.build_gid_lab_dict(wav_lst_tr + wav_lst_te)
        trainer.lab_dict = lab_dict
        trainer.data_folder = data_folder_path
        # wav_lst = wav_lst_tr + wav_lst_te
        # if args.gender == 'M' or args.gender == 'F':
        #     wav_lst = DataUtils.keep_gender(wav_lst, args.gender)
        # #train samples = 3692, #test samples = 1344
        lbls = ['M', 'F']
        tr_spk_lbls = DataUtils.get_sid_class_labels(wav_lst_tr)
        wav_lst_tr, wav_lst_val = DataUtils.distinct_gid_train_val(wav_lst_tr, tr_spk_lbls)

        # #train samples = 2960|370, #val_samples = 736|92, #test samples = 1344|168
        snt_tr = len(wav_lst_tr)
        snt_val = len(wav_lst_val)
        snt_te = len(wav_lst_te)
    elif args.dataset == 'RAVDESS':
        data_folder_path = '../../Data/RAVDESS/'
        lab_dict, wav_lst_te = DataUtils.build_gid_lab_dict_ravdess(data_folder_path)
        trainer.lab_dict = lab_dict
        trainer.data_folder = data_folder_path
        snt_te = len(wav_lst_te)
    # with open(args.save_path, 'rb') as file:
    #     trainer = CPU_Unpickler(file).load()
    # trainer.gpu = False

    # with open(args.save_path, 'rb') as file:
    #     trainer = pickle.load(file)

    trainer.wav_lst_te = wav_lst_te
    trainer.snt_te = snt_te

    trainer.CNN_net.eval()
    trainer.DNN1_net.eval()
    trainer.DNN2_net.eval()
    test_flag = 1
    loss_sum = 0
    err_sum = 0
    err_sum_snt = 0

    with torch.no_grad():
        for i in range(trainer.snt_te):
            try:
                if args.dataset == 'TIMIT' or args.dataset == 'SUB_TIMIT' or args.dataset == 'RAVDESS':
                    signal, fs = librosa.load(trainer.wav_lst_te[i], sr=16000)
                else:
                    [signal, fs] = sf.read(trainer.data_folder + trainer.wav_lst_te[i])
                ##########################################################
                if trainer.gpu:
                    signal = torch.from_numpy(signal).float().cuda().contiguous()
                else:
                    signal = torch.from_numpy(signal).float().contiguous()
                lab_batch = trainer.lab_dict[trainer.wav_lst_te[i]]
                pout, lab = trainer.split_signals_into_chunks(lab_batch, signal)
                pred = torch.max(pout, dim=1)[1]
                loss = trainer.cost(pout, lab.long())
                err = torch.mean((pred != lab.long()).float())
                [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
                ##########################################################
                if best_class.item() != lab_batch:
                    err_sum_snt += 1
                loss_sum = loss_sum + loss.detach()
                err_sum = err_sum + err.detach()
            except:
                print(trainer.wav_lst_te[i] + ' ignored for failure in openning')
                trainer.snt_te = trainer.snt_te - 1
                continue

        err_tot_dev_snt = err_sum_snt / trainer.snt_te
        loss_tot_dev = loss_sum / trainer.snt_te
        err_tot_dev = err_sum / trainer.snt_te

    with open('IO/final/_' + args.dataset + "gid_res.res", "a") as res_file:
        res_file.write(args.dataset + ' : ' + model_path + " : loss_te=%f err_te=%f err_te_snt=%f" % (loss_tot_dev, err_tot_dev, err_tot_dev_snt))

    print(args.dataset + ' : ' + model_path + " : loss_te=%f err_te=%f err_te_snt=%f" % (loss_tot_dev, err_tot_dev, err_tot_dev_snt))
    print('Done!')
