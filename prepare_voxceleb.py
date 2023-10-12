import subprocess
import numpy as np
import wave
from pydub import AudioSegment
import soundfile as sf
from Utils.


def merge(sil):
    ans = []
    check = False
    for i in range(1, len(sil)):
        tmp = sil[i - 1][1]
        if tmp == sil[i][0]:
            ans.append((sil[i - 1][0], sil[i][1]))
            check = True
        else:
            if not check:
                ans.append(sil[i - 1])
            else:
                check = False
    if ans[-1][1] != sil[-1][1]:
        ans.append(sil[-1])
    return ans


def visualize(path: str, sil=None):
    raw = wave.open(path)
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")
    f_rate = raw.getframerate()
    time = np.linspace(
        0,  # start
        len(signal) / f_rate,
        num=len(signal)
    )
    if sil:
        print(path)
        new_sig = []
        start_index = 0
        for i in sil:
            if start_index != int(i[0] * f_rate):
                new_sig.append(signal[start_index: int(i[0] * f_rate)])
            start_index = int(i[1] * f_rate)
        new_sig.append(signal[start_index:])
        new_sig = np.concatenate(new_sig).ravel()
        return new_sig, f_rate
    return signal, f_rate


def detect_silence(path, time):
    command = "ffmpeg -i " + path + " -af silencedetect=n=-30dB:d=" + str(time) + " -f null -"
    out = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    s = stdout.decode("utf-8")
    k = s.split('[silencedetect @')
    if len(k) == 1:
        # print(stderr)
        return None

    start, end = [], []
    for i in range(1, len(k)):
        x = k[i].split(']')[1]
        if i % 2 == 0:
            x = x.split('|')[0]
            x = x.split(':')[1].strip()
            end.append(float(x))
        else:
            x = x.split(':')[1]
            x = x.split('size')[0]
            x = x.replace('\r', '')
            x = x.replace('\n', '').strip()
            start.append(float(x))
    return list(zip(start, end))


data_folder_path = '../../Data/voxceleb/'
data_folder_path1 = '../../Data/voxceleb1/'


with open(male_file_list) as file:
    for line in file:
        line = line.strip()
        line = line.replace('.mp3', '.wav')
        # sound = AudioSegment.from_mp3(data_folder_path + line)

        lst = detect_silence(data_folder_path1 + line, 1)
        signal, f_rate = visualize(data_folder_path1 + line, lst)

        # signal.export(data_folder_path2 + line, format="wav")
        sf.write(data_folder_path2 + line, signal, f_rate)
with open(female_file_list) as file:
    for line in file:
        line = line.strip()
        line = line.replace('.mp3', '.wav')
        # sound = AudioSegment.from_mp3(data_folder_path + line)

        lst = detect_silence(data_folder_path1 + line, 1)
        signal, f_rate = visualize(data_folder_path1 + line, lst)

        # signal.export(data_folder_path2 + line, format="wav")
        sf.write(data_folder_path2 + line, signal, f_rate)

print('Finished!!!!')
