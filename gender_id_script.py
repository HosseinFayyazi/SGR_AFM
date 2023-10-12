import argparse
from SIDTrainer import *
import argparse
from Utils.DataUtils import *

import torch
import pickle

from SIDTrainer import *

torch.autograd.set_detect_anomaly(True)

print('Initializing parameters ...')
parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str, default='IO/final/G_custom_freeze/KernelCustomFreeze_TIMIT.cfg',
                    help='path of configs file')
parser.add_argument('--reduce_data', type=int, default='0', help='reduce data to run faster')
parser.add_argument('--resume_epoch', type=int, default='0', help='resume training from this epoch')
parser.add_argument('--resume_model_path', type=str, default='None',
                    help='resume training from the model with specified path')
parser.add_argument('--save_path', type=str, default='IO/final/G_custom_freeze/saved_model.pth',
                    help='resume training from the model with specified path')
parser.add_argument('--freeze_first_layer', type=str, default='0',
                    help='freeze first layer or not? (Yes: 1, No: 0')
parser.add_argument('--custom_filters_path', type=str, default='0',
                    help='use custom filters in first layer? (Yes: PATH, No: 0')
parser.add_argument('--dataset', type=str, default='TIMIT',
                    help='TIMIT or librispeech')

args = parser.parse_args()

print('Reading options from configs file ...')
options = GeneralUtils.read_conf(args.cfg_file)

if args.resume_model_path != 'None':
    options['pt_file'] = args.resume_model_path
# setting the seed
torch.manual_seed(int(options['seed']))
np.random.seed(int(options['seed']))

print('Retrieving train and val list ..')
lbls = ['M', 'F']

if args.dataset == 'librispeech':
    data_folder_path = '../../Data/LibriSpeech/test1/'
    gender_txt_file = '../../Data/LibriSpeech/GENDERS.txt'
    trn_file_directory = '../../Data/LibriSpeech/train1/*.wav'
    val_file_directory = '../../Data/LibriSpeech/dev1/*.wav'
    test_file_directory = '../../Data/LibriSpeech/test1/*.wav'
    wav_lst_tr, lab_dict_tr = DataUtils.build_gid_lab_dict_libri_train(trn_file_directory, gender_txt_file, 'train1')
    wav_lst_val, lab_dict_val = DataUtils.build_gid_lab_dict_libri_train(val_file_directory, gender_txt_file, 'dev1')
    wav_lst_te, lab_dict_te = DataUtils.build_gid_lab_dict_libri_train(test_file_directory, gender_txt_file, 'test1')
    lab_dict = {}
    lab_dict.update(lab_dict_tr)
    lab_dict.update(lab_dict_val)
    lab_dict.update(lab_dict_te)
    snt_tr = len(wav_lst_tr)
    snt_val = len(wav_lst_val)
    snt_te = len(wav_lst_te)

else:
    wav_lst_tr = DataUtils.read_wav_file_names(options['data_folder'], train=1)
    wav_lst_tr = DataUtils.remove_sa_wav_files(wav_lst_tr)
    wav_lst_te = DataUtils.read_wav_file_names(options['data_folder'], train=0)
    wav_lst_te_comp = DataUtils.remove_sa_wav_files(wav_lst_te)
    wav_lst_te_core = GeneralUtils.get_core_wav_test_files(wav_lst_te_comp)
    wav_lst_te = wav_lst_te_comp

    tr_spk_lbls = DataUtils.get_sid_class_labels(wav_lst_tr)
    wav_lst_tr, wav_lst_val = DataUtils.distinct_gid_train_val(wav_lst_tr, tr_spk_lbls)

    # #train samples = 2960|370, #val_samples = 736|92, #test samples = 1344|168
    snt_tr = len(wav_lst_tr)
    snt_val = len(wav_lst_val)
    snt_te = len(wav_lst_te)

     #for test
    tr_spk_lbls = DataUtils.get_sid_class_labels(wav_lst_tr)
    val_spk_lbls = DataUtils.get_sid_class_labels(wav_lst_val)
    te_spk_lbls = DataUtils.get_sid_class_labels(wav_lst_te)
    print(list(set(tr_spk_lbls) & set(val_spk_lbls)))
    print(list(set(tr_spk_lbls) & set(te_spk_lbls)))
    print(list(set(te_spk_lbls) & set(val_spk_lbls)))

    # build label dictionary
    lab_dict = DataUtils.build_gid_lab_dict(wav_lst_tr + wav_lst_val + wav_lst_te)

options['class_lay'] = str(len(lbls))
print(f'The model will be trained to identify {len(lbls)} speakers.')

# check folder exists
GeneralUtils.check_folder_exist(options['output_folder'])

print(f'Training {options["kernel_type"]} model ...')
freeze_first_layer = False
if args.freeze_first_layer == '1':
    freeze_first_layer = True
custom_filters_path = ''
if args.custom_filters_path != '0':
    custom_filters_path = args.custom_filters_path
trainer = SIDTrainer(options, wav_lst_tr, snt_tr, wav_lst_val, snt_val, lab_dict, args.save_path, freeze_first_layer, custom_filters_path)
trainer.train(int(args.resume_epoch))
with open(args.save_path, 'wb') as filehandler:
    pickle.dump(trainer, filehandler)

print('Operation completed successfully!')


