from python_speech_features import mfcc
import numpy as np
import scipy.io.wavfile as wav
from Utils.DataUtils import *
from Utils.GeneralUtils import *
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from Utils.PlotUtils import *


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
X_c_tr, y_c_tr, X_f_tr, y_f_tr = DataUtils.get_mfcc_features(wav_lst_tr, lab_dict)
X_c_val, y_c_val, X_f_val, y_f_val = DataUtils.get_mfcc_features(wav_lst_val, lab_dict, reduce_frames=False)
X_c_te, y_c_te, X_f_te, y_f_te = DataUtils.get_mfcc_features(wav_lst_te, lab_dict, reduce_frames=False)

# ---------------------------------------------------------------------------------------------------------------------#
# clf = svm.SVC(kernel='linear')
# clf = svm.SVC(kernel='poly', degree=8)
clf = svm.SVC(kernel='rbf', C=10)
# clf = svm.SVC(kernel='sigmoid')
# ---------------------------------------------------------------------------------------------------------------------#
print('Classification Error rate calculation ...')
clf.fit(X_c_tr, y_c_tr)

y_pre_val = clf.predict(X_c_val)
val_accuracy = accuracy_score(y_c_val, y_pre_val)
print('validation data CER: ' + str(1 - val_accuracy))


y_pre_tst = clf.predict(X_c_te)
tst_accuracy = accuracy_score(y_c_te, y_pre_tst)
print('Test data CER: ' + str(1 - tst_accuracy))

# ---------------------------------------------------------------------------------------------------------------------#
print('Frame Error rate calculation ...')
clf.fit(X_f_tr, y_f_tr)

y_pre_val = clf.predict(X_f_val)
val_accuracy = accuracy_score(y_f_val, y_pre_val)
print('validation data FER: ' + str(1 - val_accuracy))

print('Results on test data ...')
y_pre_tst = clf.predict(X_f_te)
tst_accuracy = accuracy_score(y_f_te, y_pre_tst)
print('Test data FER: ' + str(1 - tst_accuracy))

