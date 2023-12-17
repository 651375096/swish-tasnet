import numpy as np
import matplotlib.pyplot as plt
import math
from mir_eval.separation import bss_eval_sources

target_file="1024_35"
# target_file="800_35"
# target_file="848_35"
# target_file="640_35"
# mode_name="mix0.05-5层-1-i-cov-up-resnet-i-i1-elu-signal-noise-12best6.60723722993121e-07- 4.871535086511019e-07"
# mode_name="mix0.05-5层-elu-signal-noise-6 best-6.543180155224901e-07 "
# mode_name="mix3-0.05-5层-1-i-cov-up-resnet-i-i1-elu-signal-noise-12best1.4407668243639665e-07-1.6753857812190636e-07"
mode_name="1024-5层-elu-signal-noise-32best-1.803339005966657e-07 "
a1_name=target_file+"/0帧cf_fina1.txt"
c12d1_name=target_file+"/"+"0帧cf_finc12.txt"
n0=target_file+"/"+"0帧noise_fill_blanks.txt"
s_name="../out/"+mode_name+"/"+target_file+"/signal.txt"
n_name="../out/"+mode_name+"/"+target_file+"/noise.txt"
def drawPicture(pic_name,*data):
    '''该函数用来绘制图片

    :param pic_name: 图片保存的文件名
    :param data: 要绘制的数据
    :return: none
    '''
    num=len(data)
    #print(num)


    for i in range(num):
        y = data[i]
        plt.subplot(num, 1, i+1)  # 第num个图画在一个两行一列分割图的第i幅位置
        plt.plot(y)


    # plt.savefig("/picture/" + pic_name + ".jpg")
    plt.show()
def read_data(file1):
    data = np.loadtxt(file1)
    # data[np.where(data != 0)] = 1
    # data/=np.max(data)
    # data = data.reshape(len(data),1)

    print(data)
    print("长度",len(data))
    return data,len(data)

a1,a1_l=read_data(a1_name)
c12d1_1024,c12d1_hang=read_data(c12d1_name)
fl_1024,fl_hang=read_data(s_name)
n00,n00_l=read_data(n0)
n_fl,n_fl_l=read_data(n_name)
cd=min(a1_l,c12d1_hang,fl_hang)
cd_noise=min(a1_l,c12d1_hang,fl_hang,n00_l,n_fl_l)
# 源失真比（source-to-distortion ratio, SDR）、
# 源干扰比（source-to-interference ratio, SIR）和
# 源伪影比（source-to-artifact ratio, SAR）
def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    # src_anchor = np.stack([mix, mix], axis=0)
    src_ref=np.array(src_ref)
    src_est=np.array(src_est)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    print("sdr, sir, sar, popt",sdr, sir, sar, popt)
    # print(bss_eval_sources(src_ref, src_est))
    # sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    # avg_SDRi = ((sdr[0] - sdr0[0]) + (sdr[1] - sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    # return avg_SDRi
def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    # len_cropped = pred_src1_wav.shape[-1]
    len_cropped=cd_noise
    src1_wav = src1_wav[:len_cropped]
    src2_wav = src2_wav[:len_cropped]
    mixed_wav = mixed_wav[:len_cropped]
    pred_src1_wav =pred_src1_wav[:len_cropped]
    pred_src2_wav =pred_src2_wav[:len_cropped]
    gnsdr, gsir, gsar = np.zeros(2), np.zeros(2), np.zeros(2)
    total_len = 0
    # for i in range(2):
    sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                        np.array([pred_src1_wav, pred_src2_wav]), True)
    print("sdr, sir, sar",sdr, sir, sar)
    # sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
    #                                       np.array([mixed_wav, mixed_wav]), True)
    # nsdr = sdr - sdr_mixed
    # gnsdr += len_cropped * nsdr
    # gsir += len_cropped * sir
    # gsar += len_cropped * sar
    # total_len += len_cropped
    # gnsdr = gnsdr / total_len
    # gsir = gsir / total_len
    # gsar = gsar / total_len
    # print('GNSDR:', gnsdr)
    # print('GSIR:', gsir)
    # print('GSAR:', gsar)
# 皮尔逊相关系数
# 皮尔逊相关系数这个参数用来度量两个向量之间的相似度。
# corroef（）进行计算，皮尔逊相关系数取值从-1到+1，我们可以通过0.5+0.5*corrcoef（）来计算，将值调整归一化到0到1之间。
def pexxgxs(x,y):
    n = min(len(x), len(y))

    sum_xy = np.sum(np.sum(x * y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x * x))
    sum_y2 = np.sum(np.sum(y * y))
    # sum_xy = np.sum(x * y)
    # sum_x = np.sum(x)
    # sum_y = np.sum(y)
    # sum_x2 = np.sum(x * x)
    # sum_y2 = np.sum(y * y)
    pc = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    print("皮尔逊相关系数是", pc)

def Mse_value(x,y,bbb):
    i,mse_fl,zql=0,0,0
    while i < cd:
        aaa = abs(x[i] - y[i])
        # print(i)
        # if ((x[i] != 0 and y[i] != 0) or (x[i] == 0 and y[i] == 0)):
            # zql += 1
        mse_fl += aaa * aaa
        i += 1
    mse_fl /= cd
    # zql /= cd
    print(bbb,"均方误差", mse_fl)
    # print("准确率", zql)
    return mse_fl

def SNR_1(mix,fl):
    # S = S-np.mean(S)# 消除直流分量
    # S = S/np.max(np.abs(S))#幅值归一化
    # mean_S = (np.sum(S))/(len(S))#纯信号的平均值
    # PS = np.sum((S-mean_S)*(S-mean_S))
    # PN = np.sum((S-SN)*(S-SN))
    # snr=10*math.log((PS/PN), 10)
    # print("信噪比SNR",snr)
    # print("分离MSE")
    pn=Mse_value(fl,mix)
    # print("混合MSE")
    ps = Mse_value(mix, mix*2)
    snr = -10 * math.log((ps/pn), 10)
    print("SNR", snr)
# 功能：测量信号信噪比
# 输入S为纯信号,是一个numpy的1D张量
# 输入SN为带噪信号，是一个numpy的1D张量
# 输出snr为信噪比，单位为dB，是一个32为的float数
# 调用格式{snr=SNR_singlech(S,SN)}
def SNR_singlech(mix,c12,fl):
    # S = S-np.mean(S)# 消除直流分量
    # S = S/np.max(np.abs(S))#幅值归一化
    # mean_S = (np.sum(S))/(len(S))#纯信号的平均值
    # PS = np.sum((S-mean_S)*(S-mean_S))
    # PN = np.sum((S-SN)*(S-SN))
    # snr=10*math.log((PS/PN), 10)

    # print("信噪比SNR",snr)
    # print("分离MSE")
    ps=Mse_value(fl,c12,"分离MSE")
    # print("混合MSE")
    pn = Mse_value(mix, c12,"混合MSE")
    snr = -10 * math.log((ps/pn), 10)
    print("信噪比SNR", snr)


print(s_name)
###信噪比
# SNR_1(a1,fl_1024)
SNR_singlech(a1,c12d1_1024,fl_1024)
#MSE
# Mse_value(c12d1_1024,fl_1024)
#SDR,sir,sar

# 皮尔逊相关系数0
pexxgxs(c12d1_1024[0:cd],fl_1024[0:cd])
bss_eval_global(a1,c12d1_1024,n00,fl_1024,n_fl)
# print("\n混合信号")
# # SNR_singlech(a1,c12d1_1024,fl_1024)
# #MSE
# # Mse_value(c12d1_1024,a1)
# #SDR
# cal_SDRi(c12d1_1024[0:cd],a1[0:cd],a1[0:cd])
# # 皮尔逊相关系数
# pexxgxs(c12d1_1024,a1)
