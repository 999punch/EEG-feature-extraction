import mne
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import entropy

band = np.array([[0.5, 4],  # delta
                 [4, 8],  # theta
                 [8, 13],  # alpha
                 [13, 30],  # beta
                 [30, 100],  # gamma
                 [0.5, 100]])  # 全波段


# 对脑电数据进行高通、低通滤波和陷阱滤波的函数
def filter_eeg(eeg_matrix, sfreq, l_freq, h_freq, notch_freq):
    # eeg_matrix为脑电信号矩阵，行为通道，列为采样点
    # sfreq代表采样频率
    # l_freq,h_freq分别为高通，低通滤波器的截止频率
    # notch_freq为陷阱滤波器的频率
    # 创建一个RawArray对象
    info = mne.create_info(ch_names=['channel' + str(i) for i in range(eeg_matrix.shape[0])],
                           sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_matrix, info, copy='both')  # 此处copy='both'可使传入的参数
    # 和RawArray对象中的参数独立修改，互不影响
    # 应用高通和低通滤波器
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    # 应用陷阱滤波器
    raw.notch_filter(notch_freq, notch_widths=1)
    # 返回滤波后的数据
    return raw


# 将多维特征矩阵压缩为一维np.array数组
def compress(matrix):
    if matrix.ndim == 1:
        return matrix
    m = matrix.shape[0]
    n = matrix.shape[1]
    new_matrix = matrix.reshape(1, m * n)[0]
    return new_matrix


# 计算脑电信号各通道的香农熵的函数
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
        # 计算香农熵，加上一个极小值防止对0取对数
        SE[i] = entropy(probabilities, base=2)
        # SE[i] = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))

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
    # data为输入数据矩阵，fs为采样率，window为进行傅里叶变化的窗函数，nperseg代表窗长度，noverlap代表窗口重合度
    # nfft可理解为功率谱图的分辨率，即每隔(1/c)Hz计算一次功率谱，detrend代表去趋势化参数
    # f为频率横轴，pxx为功率谱密度，是一个通道数*频率采样点数的矩阵
    f, pxx = scipy.signal.welch(data, fs=srate, window='hann', nperseg=srate * 2, noverlap=0,
                                nfft=c * srate, detrend=False)
    # 取对数，得出相对功率谱密度
    psd = 10 * np.log10(pxx)
    # 给每个通道计算每个波段的功率谱熵（通道序号从0开始）
    for i in range(0, channels):
        for j in range(0, 6):
            f_bool = (f >= band[j][0]) & (f <= band[j][1])
            f_band = f[f_bool]
            pxx_band = pxx[i, f_bool]
            # pxx_prob = pxx_band / np.sum(pxx_band)    不需要，scipy.stats.entropy()会自动对数据进行归一化
            PSE[i, j] = entropy(pxx_band, base=2)

            # PSE[i, 0] = -np.sum(pxx[i, round(0.5 * c):4 * c] *
            #                     np.log2(pxx[i, round(0.5 * c):4 * c]))
            # PSE[i, 1] = -np.sum(pxx[i, 4 * c:8 * c] * np.log2(pxx[i, 4 * c:8 * c]))
            # PSE[i, 2] = -np.sum(pxx[i, 8 * c:13 * c] * np.log2(pxx[i, 8 * c:13 * c]))
            # PSE[i, 3] = -np.sum(pxx[i, 13 * c:30 * c] * np.log2(pxx[i, 13 * c:30 * c]))
            # PSE[i, 4] = -np.sum(pxx[i, 30 * c:100 * c] * np.log2(pxx[i, 30 * c:100 * c]))
            # PSE[i, 5] = -np.sum(pxx[i, round(0.5 * c):100 * c] *
            #                     np.log2(pxx[i, round(0.5 * c):100 * c]))

    # # 画出各通道功率谱图
    # plt.figure()
    # plt.plot(f[0:100 * c], psd[0, 0:100 * c])
    # plt.plot(f[0:100 * c], psd[1, 0:100 * c])
    # plt.plot(f[0:100 * c], psd[2, 0:100 * c])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD [V^2/Hz]')
    # plt.grid(True)
    # plt.show()

    return PSE


# 计算各波段、各通道中心频率的函数
def cf(data, srate, channels):
    # data为脑电信号矩阵，行为通道，列为采样点
    # srate代表采样频率
    # channels代表通道数
    CF = np.zeros((channels, 6))
    c = 20
    f, psd = scipy.signal.welch(data, fs=srate, window='hann', nperseg=srate * 2, noverlap=0,
                                nfft=c * srate, detrend=False)
    for i in range(0, channels):
        for j in range(0, 6):
            f_bool = (f >= band[j][0]) & (f < band[j][1])
            f_band = f[f_bool]
            psd_band = psd[i, f_bool]
            CF[i, j] = np.sum(psd_band * f_band) / np.sum(psd_band)

    return CF


# 计算各波段、各通道相对中心频率的函数
def rcf(data, srate, channels):
    # data为脑电信号矩阵，行为通道，列为采样点
    # srate代表采样频率
    # channels代表通道数
    RCF = np.zeros((channels, 6))
    CF = cf(data, srate, channels)

    for i in range(channels):
        RCF[i] = CF[i] / CF[i][-1]

    return RCF



# 用于函数编写时的测试
def main():
    mne.set_log_level(verbose='WARNING')
    data = scipy.io.loadmat(
        r"D:\IDM\下载\抑郁症数据集\MODMA数据集\EEG_3channels_resting_lanzhou_2015\02020018_still.mat")
    data = data['data'] / 1000000
    data = data.T
    data = data[:, np.arange(round(data.shape[1]/2-30*250),round(data.shape[1]/2+30*250))]
    print(data.shape)
    newraw = filter_eeg(data, 250, 0.5, 100, 50)

    # # 功率谱熵
    # p = pse(newraw.get_data(), newraw.info['sfreq'], newraw.info['nchan'])
    # print(p)
    # plt.show()

    # # 香农熵
    # s = se(newraw.get_data(), newraw.info['nchan'])
    # print(s)

    # # 中心频率
    # q = cf(newraw.get_data(), newraw.info['sfreq'], newraw.info['nchan'])
    # print(q)

    # 相对中心频率
    r = rcf(newraw.get_data(), newraw.info['sfreq'], newraw.info['nchan'])
    print(r)


if __name__ == '__main__':
    main()
