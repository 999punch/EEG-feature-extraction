import mne
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# 对脑电数据进行高通、低通滤波和陷阱滤波的函数
def filter_eeg(eeg_matrix, sfreq, l_freq, h_freq, notch_freq):
    # eeg_matrix为脑电信号矩阵，行为通道，列为采样点
    # sfreq代表采样频率
    # l_freq,h_freq分别为高通，低通滤波器的截止频率
    # notch_freq为陷阱滤波器的频率
    # 创建一个RawArray对象
    info = mne.create_info(ch_names=['channel' + str(i) for i in range(eeg_matrix.shape[0])],
                           sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_matrix, info, copy='both')  # 此处copy='both'可使传入的参数和
    # RawArray对象中的参数独立修改，互不影响
    # 应用高通和低通滤波器
    # raw.filter(l_freq=l_freq, h_freq=h_freq)
    # 应用陷阱滤波器
    raw.notch_filter(notch_freq, notch_widths=1, n_jobs=1)
    # 返回滤波后的数据
    return raw


# 计算脑电信号的香农熵
def se(data, channels):
    # data为脑电信号矩阵，行为通道，列为采样点
    # 初始化各通道的香农熵
    SE = np.zeros(channels)
    # 计算bin的数量，使用Sturges规则，即n=1+log2(N)
    num_bins = int(np.ceil(1 + np.log2(data.shape[1])))

    for i in range(0, channels):
        # 计算直方图
        counts, bin_edges = np.histogram(data[i, :], bins=num_bins)
        # 计算概率
        probabilities = counts / data.shape[1]
        print(probabilities)
        # 计算香农熵，加上一个值防止对0取对数
        SE[i] = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))

    return SE


# 计算各波段、各通道功率谱熵的函数
def pse(data, srate, channels):
    # data为脑电信号矩阵，行为通道，列为采样点
    # srate代表采样频率
    # channels代表通道数
    # c*srate表示功率谱密度图的分辨率为1/c，即每(1/c)Hz计算一次
    c = 20
    # PSE为一个记录功率谱熵的矩阵，行代表通道，列代表各个波段的功率谱熵
    PSE = np.zeros((channels, 6))
    f, pxx = scipy.signal.welch(data, fs=srate, window='hann', nperseg=srate * 2, noverlap=0,
                                nfft=c * srate, detrend=False)
    # 取对数，得出相对功率谱密度
    psd = 10 * np.log10(pxx)
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
    # plt.figure()
    # plt.plot(f[0:100 * c], psd[0, 0:100 * c])
    # plt.plot(f[0:100 * c], psd[1, 0:100 * c])
    # plt.plot(f[0:100 * c], psd[2, 0:100 * c])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD [V^2/Hz]')
    # plt.grid(True)
    # plt.show()
    return PSE


# 用于函数编写时的测试
def main():
    data = scipy.io.loadmat(
        r"D:\IDM\下载\抑郁症数据集\MODMA数据集\EEG_3channels_resting_lanzhou_2015\02010001_still.mat")
    data = data['data'] / 1000000
    data = data.T
    newraw = filter_eeg(data, 250, 0.5, 100, 50)
    p = pse(newraw.get_data(), newraw.info['sfreq'], newraw.info['nchan'])
    q = p.reshape(1, p.shape[0] * p.shape[1])[0]
    print(p)
    print(q)
    # plt.show()
    # newraw = filter_eeg(data, 250, 0.5, 100, 50)
    # s = se(newraw.get_data(), newraw.info['nchan'])
    # print(s)


if __name__ == '__main__':
    main()
