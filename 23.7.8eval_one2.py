from src.asteroid.metrics import get_metrics
import numpy as np
compute_metrics = ["si_sdr", "sar" , "sir","sdr"]

# target_name="1.1-48-4D-7E-BE-C8-65800-600.csv0.txt"
target_name="UDP10.txt"
orinal_path="/home/chenyang/wave-tasnet/voice_data/wav/tt/"
#############Tasnet文件夹
mix_path=orinal_path+"/mix/"+target_name
s1_path=orinal_path+"/s1/"+target_name
s2_path=orinal_path+"/s2/"+target_name
s3_path=orinal_path+"/s3/"+target_name
#############Tasnet文件夹
out_path="/home/chenyang/wave-tasnet/exp/tmp/out/signal/estimate/"

e1_path=out_path+"/s1/"+target_name+".txt"
e2_path=out_path+"/s2/"+target_name+".txt"
e3_path=out_path+"/s3/"+target_name+".txt"

# ###############
# out_path="/home/dcxx/chenyang/tasnet/exp/Wave-u-net—data/"
# mode_path="tasnet_ 5lay_elu_0_0_8.606072343455806e-06/"
# data_name="tasnet_test/"
# e1_path=out_path+mode_path+data_name+"signal.txt"
# e2_path=out_path+mode_path+data_name+"noise.txt"
# e3_path=out_path+mode_path+data_name+"hang.txt"
###############
mix_np=np.loadtxt(mix_path, dtype=float,unpack=True)
print(mix_np)
s1 = np.loadtxt(s1_path , dtype=float,unpack=True)
s2=np.loadtxt( s2_path , dtype=float,unpack=True)
# s3=np.loadtxt(s3_path , dtype=float,unpack=True)
e1=np.loadtxt( e1_path, dtype=float,unpack=True)
e2=np.loadtxt(e2_path , dtype=float,unpack=True)
# e3=np.loadtxt( e3_path , dtype=float,unpack=True)
sources_np=np.array([s1,s2])
est_sources_np=np.array([e1,e2])
import matplotlib.pyplot as plt


def huatu(data,name):
    y = data
    x = range(len(y))
    plt.plot(y)
    plt.title(name)
    plt.show()
huatu(-mix_np[10:10000],"mix1")

huatu(s1[10:10000],"s1")
huatu(e1[10:10000],"e1")




utt_metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=44100,
                metrics_list=compute_metrics,
            )
for key in utt_metrics:
    print(key,':', utt_metrics[key])