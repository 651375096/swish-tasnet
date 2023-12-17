#!/usr/bin/env python
# Created on 2018/12
# Author: Kaituo XU

import argparse
import json
import os
import numpy as np
import librosa


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if out_filename=="mix":

        out_filename="mix_clean"
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)



def preprocess_txtone_dir(in_dir, out_dir, out_filename, sample_rate=8000):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith('.txt'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        # samples, _ = librosa.load(wav_path, sr=sample_rate)
        print(wav_path)
        samples= np.loadtxt(wav_path,unpack=True)####np.loadtxt(r"./MUSDB18/" + "train" + "/" + sample[key],unpack=True)

        file_infos.append((wav_path, len(samples)))

    if out_filename=="mix":
        out_filename="mix_clean"
    # print()
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)

def preprocess(args):
    for data_type in ['tr']:
    # for data_type in ['tr', 'cv', 'tt']:

        for speaker in ['mix', 's1', 's2','s3']:
            #----------------------------------------------------------------------
            # 如果文件件不存在创建文件夹,则提示，并跳过
            if not os.path.exists(os.path.join(args.in_dir, data_type, speaker)):
                os.makedirs(os.path.join(args.in_dir, data_type, speaker))
                print("路径",os.path.join(args.in_dir, data_type, speaker),"不存在")
                continue
            #----------------------------------------------------------------------
            # preprocess_one_dir(os.path.join(args.in_dir, data_type, speaker),
            #                    os.path.join(args.out_dir, data_type),
            #                    speaker,
            #                    sample_rate=args.sample_rate)
            preprocess_txtone_dir(os.path.join(args.in_dir, data_type, speaker),
                               os.path.join(args.out_dir, data_type),
                               speaker,
                               sample_rate=args.sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WSJ0 data preprocessing")
    parser.add_argument('--in-dir', type=str, default='./voice_data/wav',
                        help='Directory path of wsj0 including tr, cv and tt')
    parser.add_argument('--out-dir', type=str, default='./voice_data/json',
                        help='Directory path to put output files')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Sample rate of audio file')
    args = parser.parse_args()
    print(args)
    preprocess(args)
