# -*-coding:utf-8-*-
"""
Created on 2022.3.5
programing language:python
@author:夜剑听雨
"""
import numpy as np
import math
from scipy import signal
import torch
# import tensorflow as tf
import math

import torch
import torch.nn.functional as F


def ssim(signal1, signal2, window_size=11, size_average=True):
    # Set dimensions
    u=max(signal2)
    C1 = (0.01 * u) ** 2
    C2 = (0.03 * u) ** 2
    stride = 1

    # Create a 1D Gaussian kernel
    kernel = create_gaussian_kernel(window_size)

    # Convert data types to match
    signal1 = signal1.float()
    signal2 = signal2.float()
    kernel = kernel.float()

    # Expand dimensions for convolution
    signal1 = signal1.unsqueeze(0).unsqueeze(0)
    signal2 = signal2.unsqueeze(0).unsqueeze(0)

    # Compute local mean
    mu1 = F.conv1d(signal1, kernel.unsqueeze(0).unsqueeze(0), stride=stride, padding=0)
    mu2 = F.conv1d(signal2, kernel.unsqueeze(0).unsqueeze(0), stride=stride, padding=0)

    # Compute local variance
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv1d(signal1 * signal1, kernel.unsqueeze(0).unsqueeze(0), stride=stride, padding=0) - mu1_sq
    sigma2_sq = F.conv1d(signal2 * signal2, kernel.unsqueeze(0).unsqueeze(0), stride=stride, padding=0) - mu2_sq
    sigma12 = F.conv1d(signal1 * signal2, kernel.unsqueeze(0).unsqueeze(0), stride=stride, padding=0) - mu1_mu2

    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=1)


def create_gaussian_kernel(window_size, sigma=1.5):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    result=gauss / gauss.sum()
    result = result.float()
    return result


# Example usage
# signal1 = torch.randn(256)  # Input signal 1
# signal2 = torch.randn(256)  # Input signal 2
#
# # Compute SSIM
# score = ssim(signal1, signal2)
#
# print("SSIM score:", score.item())


def normalized(origin_signal):
    y = origin_signal - np.mean(origin_signal)  # 消除直流分量
    y = y / np.max(np.abs(y))  # 幅值归一化
    return y


def compare_SNR(recov_img, real_img):
    # """
    # 计算信噪比
    # :param recov_img:重建后或者含有噪声的数据
    # :param real_img: 干净的数据
    # :return: 信噪比
    # """

    real_var = sum(real_img**2)
    noise_var=sum((recov_img-real_img)**2)
    s = 10*math.log(real_var/noise_var, 10)
    return s

def batch_snr(de_data, clean_data):
    """
    计算一个batch的平均信噪比
    :param de_data: 去噪后的数据
    :param clean_data: 干净的数据
    :return: 一个batch的平均信噪比
    """
    De_data = de_data.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    Clean_data = clean_data.data.cpu().numpy()
    SNR = 0
    for i in range(De_data.shape[0]):
        De = De_data[i, :, :].squeeze()  # 默认压缩所有为1的维度
        Clean = Clean_data[i, :, :].squeeze()
        SNR += compare_SNR(De, Clean)
    return SNR / De_data.shape[0]

def  mse1(de_data, clean_data):
    """
    :param de_data: 去噪后的数据
    :param clean_data: 干净的数据
    """
    m = np.sum((de_data - clean_data) ** 2)  # numpy可以并行运算
    m = m / m.size  # mse.size输出矩阵的元素个数
    m =  m** 0.5
    return m

def mse(de_data, clean_data):
    """
    计算一个batch的平均信噪比
    :param de_data: 去噪后的数据
    :param clean_data: 干净的数据
    :return: 一个batch的平均信噪比
    """
    De_data = de_data.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    Clean_data = clean_data.data.cpu().numpy()
    MSE = 0
    for i in range(De_data.shape[0]):
        De = De_data[i, :, :].squeeze()  # 默认压缩所有为1的维度
        Clean = Clean_data[i, :, :].squeeze()
        MSE += mse1(De, Clean)
    return MSE / De_data.shape[0]
def Psnr(signal, noise_data):
    """
    计算峰值信噪比
    Args:
        signal: 信号
        noise_data: 含噪声数据
    Returns:峰值信噪比
    """
    # signal = normalized(signal)
    # noise_data = normalized(noise_data)
    a=abs(signal.max())
    c=20000
    b=np.sum((signal - noise_data) ** 2)/c
    d=a/b
    psnr = 2 * 10 * math.log(d,10)
    return psnr


def batch_psnr(clean_data,de_data):
    """
    计算一个batch的平均信噪比
    :param de_data: 去噪后的数据
    :param clean_data: 干净的数据
    :return: 一个batch的平均信噪比
    """
    De_data = de_data.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    Clean_data = clean_data.data.cpu().numpy()
    psnr = 0
    for i in range(De_data.shape[0]):
        De = De_data[i, :, :].squeeze()  # 默认压缩所有为1的维度
        Clean = Clean_data[i, :, :].squeeze()
        Clean = np.array(Clean)
        De = np.array(De)
        psnr += Psnr(Clean,De)
    return psnr / De_data.shape[0]


def fft_spectrum(Signal, SampleRate):
    """
    计算一维信号的傅里叶谱
    :param Signal: 一维信号
    :param SampleRate: 采样率，一秒内的采样点数
    :return: 傅里叶变换结果
    """
    fft_len = Signal.size  # 傅里叶变换长度
    # 原函数值的序列经过快速傅里叶变换得到一个复数数组，复数的模代表的是振幅，复数的辐角代表初相位
    SignalFFT = np.fft.rfft(Signal) / fft_len  # 变换后归一化处理
    SignalFreqs = np.linspace(0, SampleRate/2, int(fft_len/2)+1)  # 生成频率区间
    SignalAmplitude = np.abs(SignalFFT) * 2   # 复数的模代表的是振幅
    return SignalFreqs, SignalAmplitude

# 巴沃斯低通滤波器
def butter_lowpass(cutoff, sample_rate, order=4):
    # 设置滤波器参数
    rate = sample_rate * 0.5
    normal_cutoff = cutoff / rate
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(noise_data, cutoff, sample_rate, order=4):
    """
    低通滤波器
    :param noise_data: 含噪声数据
    :param cutoff: 低通滤波的最大值
    :param sample_rate: 数据采样率
    :param order: 滤波器阶数，默认为4
    :return: 滤波后的数据
    """
    b, a = butter_lowpass(cutoff, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data

# 巴沃斯带通滤波器
def butter_bandpass(lowcut, highcut, sample_rate, order=4):
    # 设置滤波器参数
    rate = sample_rate * 0.5
    low = lowcut / rate
    high = highcut / rate
    b, a = signal.butter(order, [low, high], btype='bandpass', analog=False)
    return b, a

def bandpass_filter(noise_data, lowcut, highcut, sample_rate, order=4):
    """
    带通滤波器
    :param noise_data: 含噪声数据
    :param lowcut: 带通滤波的最小值
    :param higtcut: 带通滤波的最大值
    :param sample_rate: 数据采样率
    :param order: 滤波器阶数，默认为4
    :return: 滤波后的数据
    """
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data
# 巴沃斯高通滤波器
def butter_highpass(cutup, sample_rate, order=4):
    # 设置滤波器参数
    rate = sample_rate * 0.5
    normal_cutup = cutup / rate
    b, a = signal.butter(order, normal_cutup, btype='high', analog=False)
    return b, a

def highpass_filter(noise_data, cutup, sample_rate, order=4):
    """
    低通滤波器
    :param noise_data: 含噪声数据
    :param cutoff: 低通滤波的最大值
    :param sample_rate: 数据采样率
    :param order: 滤波器阶数，默认为4
    :return: 滤波后的数据
    """
    b, a = butter_highpass(cutup, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data

# 一维信号的中值滤波器
# python的中值滤波函数对数组的维数要求严格，打个比方你用维数为（200，1）的数组当输入，不行！
# 必须改成（200，才会给你滤波。
def mide_filter(x,kernel_size=5):
    """
    中值滤波器
    :param x: 一维信号
    :param kernel_size: 滤波器窗口，默认为5
    :return: 中值滤波后的数据
    """
    x1 = x.reshape(x.size)
    y = signal.medfilt(x1, kernel_size=kernel_size)
    return y

def fk_spectra(data, dt, dx, L=6):
    """
    f-k(频率-波数)频谱分析
    :param data: 二维的地震数据
    :param dt: 时间采样间隔
    :param dx: 道间距
    :param L: 平滑窗口
    :return: S(频谱结果), f(频率范围), k(波数范围)
    """
    data = np.array(data)
    [nt, nx] = data.shape  # 获取数据维度
    # 计算nk和nf是为了加快傅里叶变换速度,等同于nextpow2
    i = 0
    while (2 ** i) <= nx:
        i = i + 1
    nk = 4 * 2 ** i
    j = 0
    while (2 ** j) <= nt:
        j = j + 1
    nf = 4 * 2 ** j
    S = np.fft.fftshift(abs(np.fft.fft2(data, (nf, nk))))  # 二维傅里叶变换
    H1 = np.hamming(L)
    # 设置汉明窗口大小，汉明窗的时域波形两端不能到零，而海宁窗时域信号两端是零。从频域响应来看，汉明窗能够减少很近的旁瓣泄露
    H = (H1.reshape(L, -1)) * (H1.reshape(1, L))
    S = signal.convolve2d(S, H, boundary='symm', mode='same')  # 汉明平滑
    S = S[nf // 2:nf, :]
    f = np.arange(0, nf / 2, 1)
    f = f / nf / dt
    k = np.arange(-nk / 2, nk / 2, 1)
    k = k / nk / dx
    return S, k, f




def cov(X,Y,UX,UY):
    a=len(X)
    out=0
    for i in range(a):
        t=X[i,:]
        out+=(X[i,:]-UX)*(Y[i,:]-UY)
    return out






def compare_SSIM(X, Y):
    # """
    # 计算信噪比
    # :param recov_img:重建后或者含有噪声的数据
    # :param real_img: 干净的数据
    # :return: 信噪比
    # """
    K1=0.01
    K2=0.03
    L=141
    C1=(K1*L)**2
    C2=(K2*L)**2
    C3=C2/2
    a=len(X)
    X=X.reshape(2000,1)
    Y=Y.reshape(2000,1)
    UX=np.mean(X)
    UY=np.mean(Y)
    OX=np.var(X)
    OY=np.var(Y)
    OXY=cov(X,Y,UX,UY)
    l=(2*UX*UY+C1)/(UX**2+UY**2+C1)
    c = (2 * OX * OY+C2) / (OX ** 2 + OY ** 2 + C2)
    s = (OXY+C3) / (OX *OY + C3)
    ssim=l*c*s
    ssim=ssim/2000
    return ssim*100

def batch_SSIM(X, Y):
    """
    计算一个batch的平均信噪比
    :param de_data: 去噪后的数据
    :param clean_data: 干净的数据
    :return: 一个batch的平均信噪比
    """
    De_data = X.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    Clean_data = Y.data.cpu().numpy()
    Ssim = 0
    for i in range(De_data.shape[0]):
        # Clean= normalized(Clean_data)
        # De = normalized(De_data)
        De = De_data[i, :, :] # 默认压缩所有为1的维度
        Clean = Clean_data[i, :, :]
        De = torch.from_numpy(De)
        Clean = torch.from_numpy(Clean)
        De.squeeze_(0)
        Clean.squeeze_(0)
        T=1-ssim(De, Clean)
        Ssim +=T
    return Ssim / De_data.shape[0]






if __name__ == '__main__':
    # feature_data = np.load("D:\dncnn_pytorch_cici\model\data3\denoise\de_jmdx (6).lvm.npy_0.npy")
    # label_data = np.load("D:\dncnn_pytorch_cici\model\data3/red/re_jmdx (6).lvm.npy_0.npy")
    feature_data=np.loadtxt("./exp/tmp/out/signal/estimate/s1/ICMP0.txt.txt")
    label_data=np.loadtxt("./exp/tmp/out/signal/original/mix/ICMP0.txt.txt")
    feature_data = torch.from_numpy(feature_data)
    label_data = torch.from_numpy(label_data)
    # feature_data.squeeze_(2)
    # label_data.squeeze_(2)
    #
    # feature_data.squeeze_(0)
    # label_data.squeeze_(0)
    a=ssim(feature_data,label_data)
    print(a)