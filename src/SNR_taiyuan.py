import numpy as np

# 读取txt文件
data = np.loadtxt('./exp/tmp/out/signal/estimate/s1/太原PPT单通道1号tek0020ALL20.txt.txt')
data_noise = np.loadtxt('./exp/tmp/out/signal/estimate/s2/太原PPT单通道1号tek0020ALL20.txt.txt')

# 分离信号和噪声
signal = data
noise = data_noise

# 计算信号功率
signal_power = np.mean(signal**2)

# 计算噪声功率
noise_power = np.mean(noise**2)

# 计算信噪比
snr = 10*np.log10(signal_power/noise_power)

print('信噪比 SNR:', snr)