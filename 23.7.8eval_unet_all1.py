from src.asteroid.metrics import get_metrics
import numpy as np
import pandas as pd
import os
import pprint
import json

compute_metrics,series_list = ["si_sdr", "sar" , "sir","sdr"],[]

# target_name="1.1-48-4D-7E-BE-C8-65800-600.csv0.txt"
target_name="1.1-48-4D-7E-BE-C8-65800-600.csv0.txt"
orinal_path="/home/dcxx/chenyang/tasnet/voice_data/wav/tt"
#############Tasnet文件夹
out_path = "/home/dcxx/chenyang/tasnet/exp/Wave-u-net—data/"
data_name = "tasnet_test/"
mode_path = "tasnet_ 5lay_elu_0_0_8.606072343455806e-06/"

for target_name in os.listdir(orinal_path+"/mix"):
    print("target_name", target_name)
    if target_name.endswith('.txt'):
        mix_path = orinal_path + "/mix/" + target_name
        s1_path = orinal_path + "/s1/" + target_name
        s2_path = orinal_path + "/s2/" + target_name
        s3_path = orinal_path + "/s3/" + target_name
        #############Tasnet文件夹
        # e1_path=out_path+"/s1/"+target_name+".txt"
        # e2_path=out_path+"/s2/"+target_name+".txt"
        # e3_path=out_path+"/s3/"+target_name+".txt"
        ###############
        e1_path = out_path + data_name + mode_path + target_name + "signal.txt"
        e2_path = out_path + data_name + mode_path + target_name + "noise.txt"
        e3_path = out_path + data_name + mode_path + target_name + "hang.txt"
        ###############
        mix_np = np.loadtxt(mix_path, dtype=float, unpack=True)
        print(mix_np)
        s1 = np.loadtxt(s1_path, dtype=float, unpack=True)
        s2 = np.loadtxt(s2_path, dtype=float, unpack=True)
        s3 = np.loadtxt(s3_path, dtype=float, unpack=True)
        e1 = np.loadtxt(e1_path, dtype=float, unpack=True)
        e2 = np.loadtxt(e2_path, dtype=float, unpack=True)
        e3 = np.loadtxt(e3_path, dtype=float, unpack=True)
        sources_np = np.array([s1, s2, s3])
        est_sources_np = np.array([e1, e2, e3])

        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=44100,
            metrics_list=compute_metrics,
        )
        for key in utt_metrics:
            print(key, ':', utt_metrics[key])
        series_list.append(pd.Series(utt_metrics))

all_metrics_df = pd.DataFrame(series_list)
all_metrics_df.to_csv(os.path.join(out_path + data_name + mode_path, "all_metrics.csv"))


# Print and save summary metrics
final_results = {}
for metric_name in compute_metrics:
    input_metric_name = "input_" + metric_name
    ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
    final_results[metric_name] = all_metrics_df[metric_name].mean()
    final_results[metric_name + "_imp"] = ldf.mean()
print("Overall metrics :")
print("final_results",final_results)
print("final_results_end")
with open(os.path.join(out_path + data_name + mode_path, "final_metrics.json"), "w") as f:
    json.dump(final_results, f, indent=0)

for key in final_results:
    print("final_results",key, ':', final_results[key])


# for key in all_metrics_df:
#     print(key,':', all_metrics_df[key])