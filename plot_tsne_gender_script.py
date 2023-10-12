from python_speech_features import mfcc
import numpy as np
import scipy.io.wavfile as wav
from Utils.DataUtils import *
from Utils.GeneralUtils import *
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from Utils.PlotUtils import *
import matplotlib.pylab as pylab
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


params = {'legend.fontsize': 'x-small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'x-small'}
pylab.rcParams.update(params)


data_folder = '../../Data/TIMIT_NORM/'
wav_lst_tr = DataUtils.read_wav_file_names(data_folder, train=1)
wav_lst_tr = DataUtils.remove_sa_wav_files(wav_lst_tr)
wav_lst_te = DataUtils.read_wav_file_names(data_folder, train=0)
wav_lst_te_comp = DataUtils.remove_sa_wav_files(wav_lst_te)
wav_lst_te_core = GeneralUtils.get_core_wav_test_files(wav_lst_te_comp)
wav_lst_te = wav_lst_te_comp
lbls = ['M', 'F']
tr_spk_lbls = DataUtils.get_sid_class_labels(wav_lst_tr)
wav_lst_tr, wav_lst_val = DataUtils.distinct_gid_train_val(wav_lst_tr, tr_spk_lbls)
lab_dict = DataUtils.build_gid_lab_dict(wav_lst_tr + wav_lst_val + wav_lst_te)

# ---------------------------------------------------------------------------------------------------------------------#
# X_c_tr, y_c_tr, X_f_tr, y_f_tr = DataUtils.get_mfcc_features(wav_lst_tr, lab_dict)
# X_c_val, y_c_val, X_f_val, y_f_val = DataUtils.get_mfcc_features(wav_lst_val, lab_dict)
# X_c_te, y_c_te, X_f_te, y_f_te = DataUtils.get_mfcc_features(wav_lst_te, lab_dict)
# PlotUtils.plot_t_sne(X_c_te, DataUtils.set_lbl_str(y_c_te, ['Male', 'Female']),
#                      n_classes=2, title='MFCC features T-SNE projection')

file_name = 'IO/final/cnn_feats/G_custom_freeze_feats'

with open(file_name + '.pt', 'rb') as file:
    feats = CPU_Unpickler(file).load()
cnn_feats = []
lbls = []
i = 0
for i in range(feats['feats'].__len__()):
    cnn_feats.append(feats['feats'][i].cpu().detach().numpy())
    lbls.append(feats['lbls'][i].cpu().detach().numpy())
PlotUtils.plot_t_sne(cnn_feats, DataUtils.set_lbl_str(lbls, ['Male', 'Female']),
                     n_classes=2, title='MFCC features T-SNE projection')
plt.savefig(file_name + '.png', dpi=900)
plt.close()
print('OK!!!!')
