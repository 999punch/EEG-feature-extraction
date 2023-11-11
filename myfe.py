import mne
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

data = scipy.io.loadmat(r"D:\IDM\下载\抑郁症数据集\MODMA数据集\EEG_3channels_resting_lanzhou_2015\02010001_still.mat")
data = data['data'] / 1000000
data = data.T


# 计算各波段、各通道功率谱熵的函数
def pse(data, srate, channels):
    # c*srate表示功率谱密度图的分辨率为1/c，即每(1/c)Hz计算一次
    c = 20
    # PSE为一个记录功率谱熵的矩阵，行代表通道，列代表各个波段的功率谱熵
    PSE = np.zeros((channels, 6))
    f, pxx = scipy.signal.welch(data, fs=srate, window='hann', nperseg=srate * 2, noverlap=0,
                                nfft=c * srate, detrend=False)
    # 给每个通道计算每个波段的功率谱熵（通道序号从0开始）
    for i in range(0, channels):
        PSE[i, 0] = -np.sum(pxx[i, round(0.5 * c):4 * c] *
                            np.log2(pxx[i, round(0.5 * c):4 * c]))
        PSE[i, 1] = -np.sum(pxx[i, 4 * c:8 * c] * np.log2(pxx[i, 4 * c:8 * c]))
        PSE[i, 2] = -np.sum(pxx[i, 8 * c:13 * c] * np.log2(pxx[i, 8 * c:13 * c]))
        PSE[i, 3] = -np.sum(pxx[i, 13 * c:30 * c] * np.log2(pxx[i, 13 * c:30 * c]))
        PSE[i, 4] = -np.sum(pxx[i, 30 * c:100 * c] * np.log2(pxx[i, 30 * c:100 * c]))
        PSE[i, 5] = -np.sum(pxx[i, round(0.5 * c):100 * c] *
                            np.log2(pxx[i, round(0.5 * c):100 * c]))
    # 画出各通道功率谱图
    # psd = 10 * np.log10(pxx)
    # fig, ax = plt.subplots()
    # plt.plot(f[0:100 * c], psd[0, 0:100 * c])
    # plt.plot(f[0:100 * c], psd[1, 0:100 * c])
    # plt.plot(f[0:100 * c], psd[2, 0:100 * c])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD [V^2/Hz]')
    # ax.grid(True)
    # plt.show()
    return PSE


# a = pse(data, 250, 3)
# print(a)
