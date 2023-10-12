import pickle
import torch
import soundfile as sf
import argparse
import io
from Utils.GeneralUtils import *
from Utils.DataUtils import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sn


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='IO/final/G_sinc/1_saved_model.pth')
parser.add_argument('--cfg_file', type=str, default='IO/final/G_sinc/KernelSinc_TIMIT.cfg')
parser.add_argument('--gender', type=str, default='B',
                    help='gender of speaker (Male: M, Female: F, Both: B')
args = parser.parse_args()

options = GeneralUtils.read_conf(args.cfg_file)
wav_lst_tr = DataUtils.read_wav_file_names(options['data_folder'], train=1)
wav_lst_tr = DataUtils.remove_sa_wav_files(wav_lst_tr)
wav_lst_te = DataUtils.read_wav_file_names(options['data_folder'], train=0)
wav_lst_te_comp = DataUtils.remove_sa_wav_files(wav_lst_te)
wav_lst_te_core = GeneralUtils.get_core_wav_test_files(wav_lst_te_comp)
wav_lst_te = wav_lst_te_comp

lbls = ['M', 'F']
tr_spk_lbls = DataUtils.get_sid_class_labels(wav_lst_tr)
wav_lst_tr, wav_lst_val = DataUtils.distinct_gid_train_val(wav_lst_tr, tr_spk_lbls)

snt_tr = len(wav_lst_tr)
snt_val = len(wav_lst_val)
snt_te = len(wav_lst_te)

options['class_lay'] = str(len(lbls))
print(f'The model will be trained to identify {len(lbls)} speakers.')

# build label dictionary
lab_dict = DataUtils.build_gid_lab_dict(wav_lst_tr + wav_lst_val + wav_lst_te)

with open(args.save_path, 'rb') as file:
    trainer = pickle.load(file)

# with open(args.save_path, 'rb') as file:
#     trainer = CPU_Unpickler(file).load()
# trainer.gpu = False


trainer.wav_lst_te = wav_lst_te
trainer.snt_te = snt_te

trainer.CNN_net.eval()
trainer.DNN1_net.eval()
trainer.DNN2_net.eval()
test_flag = 1
loss_sum = 0
err_sum = 0
err_sum_snt = 0

feats = []
lbls = []
with torch.no_grad():
    for i in range(trainer.snt_te):
        [signal, fs] = sf.read(trainer.data_folder + trainer.wav_lst_te[i])
        pout, lab = trainer.test_one_signal1(signal, i)
        result = pout.sum(dim=0) / pout.shape[0]
        feats.append(result)
        lbls.append(lab[0])
cnn_feats = {'feats': feats, 'lbls': lbls}
with open('IO/final/cnn_feats/G_sinc_feats.pt', 'wb') as filehandler:
    pickle.dump(cnn_feats, filehandler)

print('OK!!!')
