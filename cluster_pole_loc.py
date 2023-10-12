from Utils.SignalUtils import *
from Utils.GeneralUtils import *
from Utils.PlotUtils import *
from Utils.DataUtils import *
import argparse
import matplotlib.pylab as pylab
import io
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, OPTICS, AffinityPropagation, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from collections import Counter
from Utils.CustomMFCC import *
from sklearn.metrics import accuracy_score
from sklearn import svm


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

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Sinc2', choices=
                    ['Sinc', 'Sinc2', 'Gamma', 'Gauss', 'CNN'])
parser.add_argument('--model_path', type=str, default='IO/final/G_sinc2/saved_model.pth')
parser.add_argument('--cfg_file', type=str, default='IO/final/G_sinc2/KernelSinc2_TIMIT.cfg')
parser.add_argument('--out_path', type=str, default='IO/final/imgs/G_sinc2/')

args = parser.parse_args()

print(f'Loading {args.model_name} model ...')

with open(args.model_path, 'rb') as file:
    trainer = CPU_Unpickler(file).load()
trainer.gpu = False
model = {'CNN_model_par': trainer.CNN_net.state_dict(),
                  'DNN1_model_par': trainer.DNN1_net.state_dict(),
                  'DNN2_model_par': trainer.DNN2_net.state_dict()
                  }
# model = torch.load(args.model_path, map_location=torch.device('cpu'))
fs = 16000

options = GeneralUtils.read_conf(args.cfg_file)
N = int(list(map(int, options['cnn_len_filt'].split(',')))[0])
N_filt = int(list(map(int, options['cnn_N_filt'].split(',')))[0])
num_scales = 4

h_n_s, time_domain_filters, freq_domain_filters_db, phase_of_filters, freq_centers, f1_list, f2_list, amp_list = \
    SignalUtils.get_learned_filters_test(args.model_name, model, fs, N, num_scales)

# ----------------------------------------------- CONVERSION --------------------------------------------------------- #
if args.model_name == 'Gauss_Cascade':
    f_new = f1_list + f2_list
    f1_list = list(np.asarray(f_new)[:, 0])
    f2_list = list(np.asarray(f_new)[:, 1])
xy = []
w0s = []
w0_sigmas = []
for i in range(f1_list.__len__()):
    # if args.model_name == 'Gauss_Cascade':
    #     w0 = 2 * np.pi * (f1_list[i][0] + f1_list[i][1]) / (2 * fs)
    #     sigma = 2 * np.pi * (f1_list[i][1] - f1_list[i][0]) / (1 * fs)
    #     x = np.exp(-sigma + 1j * w0)
    #     xy.append([np.real(x), np.imag(x)])
    #
    #     w0 = 2 * np.pi * (f2_list[i][0] + f2_list[i][1]) / (2 * fs)
    #     sigma = 2 * np.pi * (f2_list[i][1] - f2_list[i][0]) / (1 * fs)
    #     x = np.exp(-sigma + 1j * w0)
    #     xy.append([np.real(x), np.imag(x)])
    # else:
    w0 = 2 * np.pi * (f1_list[i] + f2_list[i]) / (2 * fs)
    w0s.append(w0)
    sigma = 2 * np.pi * (f2_list[i] - f1_list[i]) / (1 * fs)
    w0_sigmas.append([w0, sigma])
    x = np.exp(-sigma + 1j * w0)
    xy.append([np.real(x), np.imag(x)])

# ----------------------------------------------- CLUSTERING --------------------------------------------------------- #
# db = MeanShift(bandwidth=0.044).fit(xy)  # 80 filters for gauss_cascade
# db = MeanShift(bandwidth=0.0736).fit(xy)  # 40 filters for gauss_cascade
db = MeanShift(bandwidth=0.079).fit(xy)  # 40 filters for gauss_cascade
# db = MeanShift(bandwidth=0.000000000000079).fit(xy)  # 40 filters for gauss_cascade


# db = AffinityPropagation(preference=-0.0216).fit(xy)
# db = OPTICS(min_samples=2).fit(xy)

# db = SpectralClustering(n_clusters=80).fit(np.asarray(w0s).reshape(-1, 1))

# db = KMeans(n_clusters=80).fit(np.asarray(w0s).reshape(-1, 1))
# db = KMeans(n_clusters=80).fit(xy)

# db = GaussianMixture(n_components=100).fit(np.asarray(w0s).reshape(-1, 1))
# labels = db.predict(np.asarray(w0s).reshape(-1, 1))

# db = DBSCAN(eps=0.01, min_samples=2).fit(np.asarray(w0s).reshape(-1, 1))
# db = DBSCAN(eps=0.05, min_samples=2).fit(xy)
# db = DBSCAN(eps=0.09, min_samples=2).fit(w0_sigmas)
# db = DBSCAN(eps=0.064, min_samples=2).fit(xy) # cascade
labels = db.labels_

counts = Counter(labels)

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
n_filters_ = n_clusters_ + n_noise_

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Estimated number of filters: %d" % n_filters_)

# ------------------------------------------------ PLOTTING ---------------------------------------------------------- #
filters_xy = []
filters_f1_f2 = []
filters_abs = []
unique_labels = list(set(labels))
f1 = []
f2 = []
freqs = []
for lbl in unique_labels:
    cluster_member = [i for i, e in enumerate(labels) if e == lbl]
    xy_ = [xy[index] for index in cluster_member]
    f1_f2_ = [[f1_list[index], f2_list[index]] for index in cluster_member]
    if lbl != -1:
        cluster_center = np.average(xy_, axis=0)
        filters_xy.append(complex(cluster_center[0], cluster_center[1]))
        f1_f2_center = np.average(f1_f2_, axis=0)
        f1.append(f1_f2_center[0])
        f2.append(f1_f2_center[1])
        freqs.append((f1_f2_center[0] + f1_f2_center[1]) / 2)
        filters_f1_f2.append(f1_f2_center)
        filters_abs.append(np.abs(complex(cluster_center[0], cluster_center[1])))
    else:
        for i in range(xy_.__len__()): #  xy_element in xy_:
            filters_xy.append(complex(xy_[i][0], xy_[i][1]))
            filters_f1_f2.append(np.asarray(f1_f2_[i]))
            f1.append(f1_f2_[i][0])
            f2.append(f1_f2_[i][1])
            freqs.append((f1_f2_[i][0] + f1_f2_[i][1]) / 2)
            filters_abs.append(np.abs(complex(xy_[i][0], xy_[i][1])))

with open(args.out_path + 'f1.pth', 'wb') as filehandler:
    pickle.dump(f1, filehandler)
with open(args.out_path + 'f2.pth', 'wb') as filehandler:
    pickle.dump(f2, filehandler)
with open(args.out_path + 'freqs.pth', 'wb') as filehandler:
    pickle.dump(freqs, filehandler)
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
for filter in filters_xy:
    w0 = np.angle(filter)
    ax.scatter(w0, abs(filter), c='b', marker='x', alpha=0.5)
    ax.scatter(-w0, abs(filter), c='r', marker='x', alpha=0.5)

xT = plt.xticks()[0]
xL = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
      r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
plt.xticks(xT, xL)
# ax.grid(False)
# ax.set_thetamin(0)
# ax.set_thetamax(180)
plt.title('Distribution of poles of learnt filters in complex plane')
plt.savefig(args.out_path + '_reduced_poles.png', dpi=900)
plt.close()


# ---------------------------------------------------- MFCC ---------------------------------------------------------- #
# audio_path = 'IO/test/SA1.wav'
# cus_mfcc = CustomMFCC()
# cc = cus_mfcc.do(audio_path, hop_size=10, FFT_size=3200, mel_filter_num=80, filters_loc=filters_xy)

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
# X_c_tr, y_c_tr, X_f_tr, y_f_tr = DataUtils.get_mfcc_features(wav_lst_tr, lab_dict, reduce_frames=True,
#                                         filters_loc=filters_xy, filters_f1_f2=filters_f1_f2, filters_abs=filters_abs)
# X_c_val, y_c_val, X_f_val, y_f_val = DataUtils.get_mfcc_features(wav_lst_val, lab_dict, reduce_frames=True,
#                                         filters_loc=filters_xy, filters_f1_f2=filters_f1_f2, filters_abs=filters_abs)
X_c_te, y_c_te, X_f_te, y_f_te = DataUtils.get_mfcc_features(wav_lst_te, lab_dict, reduce_frames=True,
                                        filters_loc=filters_xy, filters_f1_f2=filters_f1_f2, filters_abs=filters_abs)

PlotUtils.plot_t_sne(X_c_te, DataUtils.set_lbl_str(y_c_te, ['Male', 'Female']),
                     n_classes=2, title='MFCC features T-SNE projection')
plt.show()
# exit()

# # ---------------------------------------------------------------------------------------------------------------------#
# # clf = svm.SVC(kernel='linear')
# # clf = svm.SVC(kernel='poly', degree=8)
# clf = svm.SVC(kernel='rbf', C=10)
# # clf = svm.SVC(kernel='sigmoid')
# # ---------------------------------------------------------------------------------------------------------------------#
# print('Classification Error rate calculation ...')
# clf.fit(X_c_tr, y_c_tr)
#
# y_pre_val = clf.predict(X_c_val)
# val_accuracy = accuracy_score(y_c_val, y_pre_val)
# print('validation data CER: ' + str(1 - val_accuracy))
#
#
# y_pre_tst = clf.predict(X_c_te)
# tst_accuracy = accuracy_score(y_c_te, y_pre_tst)
# print('Test data CER: ' + str(1 - tst_accuracy))
#
# # ---------------------------------------------------------------------------------------------------------------------#
# print('Frame Error rate calculation ...')
# clf.fit(X_f_tr, y_f_tr)
#
# y_pre_val = clf.predict(X_f_val)
# val_accuracy = accuracy_score(y_f_val, y_pre_val)
# print('validation data FER: ' + str(1 - val_accuracy))
#
# print('Results on test data ...')
# y_pre_tst = clf.predict(X_f_te)
# tst_accuracy = accuracy_score(y_f_te, y_pre_tst)
# print('Test data FER: ' + str(1 - tst_accuracy))