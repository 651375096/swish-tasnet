from src.asteroid.metrics import get_metrics
import numpy as np
compute_metrics = ["si_sdr", "sar" , "sir","sdr"]

# out_path = "/root/autodl-tmp/chenyang/tasnet/exp/tmp/out/signal/estimate"
#
#
# mix_np=np.loadtxt(out_path, dtype=float,unpack=True)
# s1 = np.loadtxt(out_path , dtype=float,unpack=True)
# s2=np.loadtxt( out_path , dtype=float,unpack=True)
# s3=np.loadtxt(out_path , dtype=float,unpack=True)
# e1=np.loadtxt( out_path, dtype=float,unpack=True)
# e2=np.loadtxt(out_path , dtype=float,unpack=True)
# e3=np.loadtxt( out_path , dtype=float,unpack=True)
# sources_np=[s1,s2,s3]
# est_sources_np=[e1,e2,e3]

mix_np=np.array([2,3,4,5,6])
sources_np=np.array([[1,3,4,5,6],[1,3,4,5,6],[1,3,4,5,6]])
est_sources_np=np.array([[5,3,4,5,6],[1,3,4,5,6],[1,3,4,5,6]])


utt_metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=44100,
                metrics_list=compute_metrics,
            )
print("utt_metrics ",utt_metrics )