from src.asteroid.metrics import get_metrics
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import math
from mir_eval.separation import bss_eval_sources

compute_metrics = ["si_sdr", "sar" , "sir","sdr"]

# target_name="1.1-48-4D-7E-BE-C8-65800-600.csv0.txt"
target_name="1.1-48-4D-7E-BE-C8-65800-600.csv0.txt"
orinal_path="/home/dcxx/chenyang/tasnet/voice_data/wav/tt"
#############Tasnet文件夹
mix_path=orinal_path+"/mix/"+target_name
s1_path=orinal_path+"/s1/"+target_name
s2_path=orinal_path+"/s2/"+target_name
s3_path=orinal_path+"/s3/"+target_name
#############Tasnet文件夹

# e1_path=out_path+"/s1/"+target_name+".txt"
# e2_path=out_path+"/s2/"+target_name+".txt"
# e3_path=out_path+"/s3/"+target_name+".txt"

###############
out_path="/home/dcxx/chenyang/tasnet/exp/Wave-u-net—data/"
mode_path="tasnet_ 5lay_elu_0_0_8.606072343455806e-06/"
data_name="tasnet_test/"
e1_path=out_path+mode_path+data_name+"signal.txt"
e2_path=out_path+mode_path+data_name+"noise.txt"
e3_path=out_path+mode_path+data_name+"hang.txt"
###############
mix_np=np.loadtxt(mix_path, dtype=float,unpack=True)
print(mix_np)
s1 = np.loadtxt(s1_path , dtype=float,unpack=True)
s2=np.loadtxt( s2_path , dtype=float,unpack=True)
s3=np.loadtxt(s3_path , dtype=float,unpack=True)
e1=np.loadtxt( e1_path, dtype=float,unpack=True)
e2=np.loadtxt(e2_path , dtype=float,unpack=True)
e3=np.loadtxt( e3_path , dtype=float,unpack=True)
sources_np=np.array([s1,s2,s3])
est_sources_np=np.array([e1,e2,e3])


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

utt_metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=44100,
                metrics_list=compute_metrics,
            )
for key in utt_metrics:
    print(key,':', utt_metrics[key])