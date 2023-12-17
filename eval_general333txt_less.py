import os
import random
import soundfile as sf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # 按照PCI BUS ID顺序从0开始排列GPU设备
#os.environ["CUDA_VISIBLE_DEVICES"]="2"  # 设置当前程序仅使用第0、1块GPU 运行
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"  # 设置当前程序仅使用第0、1块GPU 运行
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import pdb
##########
import asteroid
# from asteroid.metrics import get_metrics
# from asteroid.data.librimix_dataset import LibriMix
# from asteroid.data.wsj0_mix import Wsj0mixDataset
# from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
# from asteroid.models import save_publishable
# from asteroid.utils import tensors_to_device
##############

from src.asteroid.metrics import get_metrics
from src.asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from src.asteroid.models import save_publishable
from src.asteroid.utils import tensors_to_device


from src.data import make_test_dataset
from src.data import make_test_dataset3
from src.models import *
#-----------begin-----------------
import soundfile
#------------end-----------------

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", default="wsj0-mix", choices=["LibriMix", "wsj0-mix"])
parser.add_argument("--model", default="ConvTasNet", choices=["ConvTasNet", "DPRNNTasNet", "DPTNet"])
parser.add_argument("--test_dir", type=str,default="./voice_data/json/tt" , help="Test directory including the csv files")
parser.add_argument("--task", type=str, default="sep_clean", choices=["sep_clean", "sep_noisy"])
parser.add_argument("--use_gpu", type=int, default=1, help="Whether to use the GPU for model execution")
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument("--out_dir", type=str, default="out", help="Directory in exp_dir where the eval results will be stored")
parser.add_argument("--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all")

#-------------------------------begin-------------------------------------------
#auto add model name
args = parser.parse_args()
model_folder="/checkpoints/"
model_name = os.listdir(args.exp_dir+model_folder)
get_key = lambda i: int((i.split('-')[0]).split('=')[1])
model_name_reverse = sorted(model_name, key=get_key,reverse=True)
model_dir="."+model_folder+model_name_reverse[0]
print("model_dir",model_dir)

print("use model  "+ model_name_reverse[0])
#--------------------------------end------------------------------------------

parser.add_argument("--ckpt_path", default=model_dir, help="Experiment checkpoint path")
parser.add_argument("--publishable", action="store_true", help="Save publishable.")

# compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]
compute_metrics = ["si_sdr", "sar" , "sir","sdr"]

#-----------------------------begin----------------------------------------
def makeDirs(*dirs):
    for dir in dirs:
        print("dir",dir)
        os.makedirs(dir, exist_ok=True)


def writeData(data, ways,fileDir,fileName):  # 将信号保存到指定路径
    '''该函数用来写入数据

    :param data:需要写入的数据
    :param ways:写入方式：txt 、 wav、lvm、lvm_head
    :param fileDir:需要写入的路径
    :param fileName:需要写入的文件名
    :param lvm_head_path:  当选择写入的文件为lvm文件时，需要填入lvm头文件地址，默认为"./data/lvm_head.lvm"
    :return result: 返回是否成功
    '''
    result = True

    #如果文件件不存在创建文件夹
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)
        print("fileDir",fileDir)

    temp_path = fileDir + fileName

    if(ways=='wav'):
        path = temp_path +'.'+ways
        soundfile.write(path, data, 44100, format='wav', subtype='PCM_16')
    elif(ways=='txt'):
        path = temp_path + '.' + ways
        np.savetxt(path, data ,fmt='%.05f')
    elif (ways == 'lvm'):
        #生成lvm数据文件
        path = temp_path + '.' + ways
        np.savetxt(path, data, fmt='%.05f')
    else:
        result = False

    return result


def getNameFromPath(path):

    #获取文件名和后缀名
    name_suffix = path.split('/')[-1]
    # split_name_suffix = name_suffix.split('.')
    split_name_suffix = name_suffix.split('.wav')

    name = split_name_suffix[0]   #文件名
    suffix = split_name_suffix[0] #后缀名

    result ={"name":name,"suffix":suffix}

    return result
#----------------------------end-----------------------------------------


def main(conf):
    print("conf",conf)
    # pdb.set_trace()
    model_path = os.path.join(conf["exp_dir"], conf["ckpt_path"])
    print("model_path",model_path)
    # all resulting files would be saved in eval_save_dir
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    os.makedirs(eval_save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(eval_save_dir, "final_metrics.json")):
        if conf["ckpt_path"] == "best_model.pth":
            # serialized checkpoint
            model = getattr(asteroid, conf["model"]).from_pretrained(model_path)
        else:
            # non-serialized checkpoint, _ckpt_epoch_{i}.ckpt, keys would start with
            # "model.", which need to be removed
            model = getattr(asteroid, conf["model"])(**conf["train_conf"]["filterbank"], **conf["train_conf"]["masknet"])
            all_states = torch.load(model_path, map_location="cpu")
            state_dict = {k.split('.', 1)[1]: all_states["state_dict"][k] for k in all_states["state_dict"]}
            model.load_state_dict(state_dict)
            # model.load_state_dict(all_states["state_dict"], strict=False)

        # Handle device placement
        if conf["use_gpu"]:
            model.cuda()
        model_device = next(model.parameters()).device
        test_set = make_test_dataset3(
            corpus=conf["corpus"],#('wsj0-mix',)
            test_dir=conf["test_dir"],# ('./voice_data/json/tt',)
            task=conf["task"],#('sep_clean',)
            sample_rate=conf["sample_rate"],
            n_src=conf["train_conf"]["data"]["n_src"],# (3,)
            )
        # Used to reorder sources only
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        # Randomly choose the indexes of sentences to save.
        ex_save_dir = os.path.join(eval_save_dir, "examples/")
        if conf["n_save_ex"] == -1:
            conf["n_save_ex"] = len(test_set)

        # 原来代码的，随机选取10个进行评价
        #save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
        #----------------------begin----------------------------------------

        save_idx = list(range(len(test_set)))
        print("save_idx ",save_idx )



        signal_save_dir = os.path.join(eval_save_dir, "signal/")
        original_save_dir = os.path.join(signal_save_dir, "original/")
        original_mix_save_dir = os.path.join(original_save_dir, "mix/")
        original_s1_save_dir = os.path.join(original_save_dir, "s1/")
        original_s2_save_dir = os.path.join(original_save_dir, "s2/")
        original_s3_save_dir = os.path.join(original_save_dir, "s3/")

        estimate_save_dir = os.path.join(signal_save_dir, "estimate/")
        estimate_s1_save_dir = os.path.join(estimate_save_dir, "s1/")
        estimate_s2_save_dir = os.path.join(estimate_save_dir, "s2/")
        estimate_s3_save_dir = os.path.join(estimate_save_dir, "s3/")

        # makeDirs(
        #     original_mix_save_dir,
        #     original_s1_save_dir,
        #     original_s2_save_dir,
        #     estimate_s1_save_dir,
        #     estimate_s2_save_dir
        #          )

        # -----------------------end---------------------------------------



        series_list = []
        torch.no_grad().__enter__()
        print("长度",len(test_set))
        # test_set < src.asteroid.data.wham_dataset.WhamDataset
        # object
        # at
        # 0x7f2a7a2f6210 >
        for idx in tqdm(range(len(test_set))):
            # Forward the network on the mixture.

            # print("test_set[idx]",len(test_set),idx,test_set[idx])

            mix, sources = tensors_to_device(test_set[idx], device=model_device)
            est_sources = model(mix.unsqueeze(0))

            #print(sources)
            #print(test_set.mix[0][0])

            # When inferencing separation for multi-task training,
            # exclude the last channel. Does not effect single-task training
            # models (from_scratch, pre+FT).
            est_sources = est_sources[:, :sources.shape[0]]

            # print("在这里",est_sources)

            loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
            mix_np = mix.cpu().data.numpy()
            sources_np = sources.cpu().data.numpy()
            est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
            # For each utterance, we get a dictionary with the mixture path,
    #########
            # the input and output metrics
            # print("mix_np",mix_np)
            # print("sources_np",sources_np,sources_np.shape)
            # print("est_sources_np",est_sources_np)
###########    # print('Is a all zeros?: ', np.all(est_sources_np[2] == 0))

            print("sources_np",sources_np)
            utt_metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=conf["sample_rate"],
                metrics_list=compute_metrics,
            )


            if hasattr(test_set, "mixture_path"):
                utt_metrics["mix_path"] = test_set.mixture_path
            series_list.append(pd.Series(utt_metrics))


            # Save some examples in a folder. Wav files and metrics as text.
            if idx in save_idx:
                print("idx ",idx)
                local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))

                os.makedirs(local_save_dir, exist_ok=True)
                print("local_save_dir ", local_save_dir)
                # sf.write(local_save_dir + "mixture.wav", mix_np, conf["sample_rate"])
                #------------------------------begin------------------------------------------
                sf.write(original_mix_save_dir + str(idx)+".wav", mix_np, conf["sample_rate"])
                #print(mix_np[0])
                mix_name = getNameFromPath(test_set.mix[idx][0])["name"]
                # writeData(mix_np,"txt",original_mix_save_dir,mix_name)

                #my_dir = conf["test_dir"]
                #print(my_dir)
                #---------------------------------end--------------------------------------
############
                # Loop over the sources and estimates
                # print("sources_np",sources_np)
                for src_idx, src in enumerate(sources_np):
                    print("src_idx, src",src_idx, src)
                    # sf.write(local_save_dir + "s{}.wav".format(src_idx), src, conf["sample_rate"])

                    #----------------------------begin--------------------------------------------
                    if src_idx == 0:
                        # aa = getNameFromPath(test_set.sources[0][idx])["name"]
                        # print(aa)
                        # print("test_set",test_set)
                        # print("test_set.sources",test_set.sources)
                        # print("test_set.sources[0]", test_set.sources[0])
                        # aa1 = getNameFromPath(test_set.sources[0][idx][0])["name"]
                        # print("aa1",aa1)
                        # aa2 = getNameFromPath(test_set.sources[1][idx][0])["name"]
                        # print(aa2)
                        #
                        # # pdb.set_trace()


                        #sf.write(original_s1_save_dir + str(idx)+".wav", src, conf["sample_rate"])
                        s1_name = getNameFromPath(test_set.sources[0][idx][0])["name"]
                        writeData(src,"txt", original_s1_save_dir, s1_name)


                    elif src_idx == 1:
                        #sf.write(original_s2_save_dir + str(idx)+".wav", src, conf["sample_rate"])
                        s2_name = getNameFromPath(test_set.sources[1][idx][0])["name"]
                        # writeData(sr  c,"txt", original_s2_save_dir, s2_name)
                    elif src_idx == 2:
                        #sf.write(original_s2_save_dir + str(idx)+".wav", src, conf["sample_rate"])
                        s3_name = getNameFromPath(test_set.sources[2][idx][0])["name"]
                        # writeData(src,"txt", original_s3_save_dir, s3_name)
                    #-----------------------------end------------------------------------------------
################
                for src_idx, est_src in enumerate(est_sources_np):
                    est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
                    # sf.write(
                    #     local_save_dir + "s{}_estimate.wav".format(src_idx),
                    #     est_src,
                    #     conf["sample_rate"],
                    # )

                    #--------------------begin------------------------------------------
                    # print("src_idx",src_idx,s1_name,s2_name,s3_name)###src_idx 2 1 1 1
                    if src_idx == 0:
                        # sf.write(
                        #     estimate_s1_save_dir + str(idx) + ".wav",
                        #     est_src,
                        #     conf["sample_rate"],
                        # )
                        writeData(est_src, "txt", estimate_s1_save_dir, s1_name)
                    elif src_idx == 1:
                        # sf.write(
                        #     estimate_s2_save_dir + str(idx) + ".wav",
                        #     est_src,
                        #     conf["sample_rate"],
                        # )
                        writeData(est_src, "txt", estimate_s2_save_dir, s2_name)
                        # print(1)
                    elif src_idx == 2:
                        # sf.write(
                        #     estimate_s2_save_dir + str(idx) + ".wav",
                        #     est_src,
                        #     conf["sample_rate"],
                        # )
                        writeData(est_src, "txt", estimate_s3_save_dir, s3_name)
                #     #----------------------end-----------------------------------------------
                # Write local metrics to the example folder.
                with open(local_save_dir + "metrics.json", "w") as f:
                    json.dump(utt_metrics, f, indent=0)

        # Save all metrics to the experiment folder.
        all_metrics_df = pd.DataFrame(series_list)
        all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

        # Print and save summary metrics
        final_results = {}
        for metric_name in compute_metrics:
            input_metric_name = "input_" + metric_name
            ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
            final_results[metric_name] = all_metrics_df[metric_name].mean()
            final_results[metric_name + "_imp"] = ldf.mean()
        print("Overall metrics :")
        pprint(final_results)
        print("final_results_end")
        with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
            json.dump(final_results, f, indent=0)
    else:
        with open(os.path.join(eval_save_dir, "final_metrics.json"), "r") as f:
            final_results = json.load(f)

    if conf["publishable"]:
        assert conf["ckpt_path"] == "best_model.pth"
        model_dict = torch.load(model_path, map_location="cpu")
        os.makedirs(os.path.join(conf["exp_dir"], "publish_dir"), exist_ok=True)
        dad=os.path.join(conf["exp_dir"], "publish_dir")
        print("319行dad",dad)
        publishable = save_publishable(
            os.path.join(conf["exp_dir"], "publish_dir"),
            model_dict,
            metrics=final_results,
            train_conf=train_conf,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )

    main(arg_dic)
