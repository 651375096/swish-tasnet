import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf
from .wsj0_mix import wsj0_license
import pdb
import matplotlib.pyplot as plt
import yaml
import pdb
DATASET = "WHAM"
# WHAM tasks

with open('./local/ConvTasNet.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(config['data']['n_src'])

if config['data']['n_src']==1:
    enh_single = {"mixture": "mix_single", "sources": ["s1"], "infos": ["noise"], "default_nsrc": 1}
    enh_both = {"mixture": "mix_both", "sources": ["mix_clean"], "infos": ["noise"], "default_nsrc": 1}
    sep_clean = {"mixture": "mix_clean", "sources": ["s1"], "infos": [], "default_nsrc": 1}
    sep_noisy = {"mixture": "mix_both", "sources": ["s1"], "infos": ["noise"], "default_nsrc": 1}
elif config['data']['n_src']==2:
    enh_single = {"mixture": "mix_single", "sources": ["s1"], "infos": ["noise"], "default_nsrc": 1}
    enh_both = {"mixture": "mix_both", "sources": ["mix_clean"], "infos": ["noise"], "default_nsrc": 1}
    sep_clean = {"mixture": "mix_clean", "sources": ["s1", "s2"], "infos": [], "default_nsrc": 2}
    sep_noisy = {"mixture": "mix_both", "sources": ["s1", "s2"], "infos": ["noise"], "default_nsrc": 2}
elif config['data']['n_src']==3:
    enh_single = {"mixture": "mix_single", "sources": ["s1"], "infos": ["noise"], "default_nsrc": 1}
    enh_both = {"mixture": "mix_both", "sources": ["mix_clean"], "infos": ["noise"], "default_nsrc": 1}
    sep_clean = {"mixture": "mix_clean", "sources": ["s1", "s2", "s3"], "infos": [], "default_nsrc": 3}
    sep_noisy = {"mixture": "mix_both", "sources": ["s1", "s2", "s3"], "infos": ["noise"], "default_nsrc": 3}

WHAM_TASKS = {
    "enhance_single": enh_single,
    "enhance_both": enh_both,
    "sep_clean": sep_clean,
    "sep_noisy": sep_noisy,
}
# Aliases.
WHAM_TASKS["enh_single"] = WHAM_TASKS["enhance_single"]
WHAM_TASKS["enh_both"] = WHAM_TASKS["enhance_both"]

def huatu(data,name):
    y = data
    x = range(len(y))
    plt.plot(y)
    plt.title(name)
    plt.show()

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)



class WhamtxtDataset(data.Dataset):
    """Dataset class for WHAM source separation and speech enhancement tasks.

    Args:
        json_dir (str): The path to the directory containing the json files.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'``.

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        nondefault_nsrc (int, optional): Number of sources in the training
            targets.
            If None, defaults to one for enhancement tasks and two for
            separation tasks.
        normalize_audio (bool): If True then both sources and the mixture are
            normalized with the standard deviation of the mixture.

    References
        "WHAM!: Extending Speech Separation to Noisy Environments",
        Wichern et al. 2019
    """

    dataset_name = "WHAM"

    def __init__(
        self,
        json_dir,
        task,
        sample_rate=8000,
        segment=4.0,
        nondefault_nsrc=None,
        normalize_audio=False,
    ):
        super(WhamDataset, self).__init__()
        if task not in WHAM_TASKS.keys():
            raise ValueError(
                "Unexpected task {}, expected one of " "{}".format(task, WHAM_TASKS.keys())
            )
        # Task setting
        self.json_dir = json_dir
        self.task = task
        self.task_dict = WHAM_TASKS[task]
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.seg_len = None if segment is None else int(segment * sample_rate)
        self.EPS = 1e-8
        if not nondefault_nsrc:
            self.n_src = self.task_dict["default_nsrc"]
        else:
            assert nondefault_nsrc >= self.task_dict["default_nsrc"]
            self.n_src = nondefault_nsrc
        self.like_test = self.seg_len is None
        # Load json files
        mix_json = os.path.join(json_dir, self.task_dict["mixture"] + ".json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in self.task_dict["sources"]
        ]
        # sources_json['./voice_data/json/tr/s1.json', './voice_data/json/tr/s2.json', './voice_data/json/tr/s3.json']
        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))
        # Filter out short utterances only when segment is specified
        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0
        if not self.like_test:
            for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
                if mix_infos[i][1] < self.seg_len:
                    drop_utt += 1
                    drop_len += mix_infos[i][1]
                    del mix_infos[i]
                    for src_inf in sources_infos:
                        del src_inf[i]

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / sample_rate / 36000, orig_len, self.seg_len
            )
        )
        self.mix = mix_infos
        # Handle the case n_src > default_nsrc
        while len(sources_infos) < self.n_src:
            sources_infos.append([None for _ in range(len(self.mix))])
        self.sources = sources_infos

    def __add__(self, wham):
        if self.n_src != wham.n_src:
            raise ValueError(
                "Only datasets having the same number of sources"
                "can be added together. Received "
                "{} and {}".format(self.n_src, wham.n_src)
            )
        if self.seg_len != wham.seg_len:
            self.seg_len = min(self.seg_len, wham.seg_len)
            print(
                "Segment length mismatched between the two Dataset"
                "passed one the smallest to the sum."
            )
        self.mix = self.mix + wham.mix
        self.sources = [a + b for a, b in zip(self.sources, wham.sources)]

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Random start
        if self.mix[idx][1] == self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len
####################################
       # Load mixture
        x, _  = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype="float32")
        print(x)
        # pdb.set_trace()
        seg_len = torch.as_tensor([len(x)])
########## Load sources
        source_arrays = []
        for src in self.sources:
            if src[idx] is None:
                # Target is filled with zeros if n_src > default_nsrc
                s = np.zeros((seg_len,))
            else:
                s, _ = sf.read(src[idx][0], start=rand_start, stop=stop, dtype="float32")
            source_arrays.append(s)
        sources = torch.from_numpy(np.vstack(source_arrays))
        mixture = torch.from_numpy(x)

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)
        return mixture, sources

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = self.task
        if self.task == "sep_clean":
            data_license = [wsj0_license]
        else:
            data_license = [wsj0_license, wham_noise_license]
        infos["licenses"] = data_license
        return infos


class WhamDataset(data.Dataset):
    """Dataset class for WHAM source separation and speech enhancement tasks.

    Args:
        json_dir (str): The path to the directory containing the json files.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'``.

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        nondefault_nsrc (int, optional): Number of sources in the training
            targets.
            If None, defaults to one for enhancement tasks and two for
            separation tasks.
        normalize_audio (bool): If True then both sources and the mixture are
            normalized with the standard deviation of the mixture.

    References
        "WHAM!: Extending Speech Separation to Noisy Environments",
        Wichern et al. 2019
    """

    dataset_name = "WHAM"

    def __init__(
        self,
        json_dir,
        task,
        sample_rate=8000,
        segment=4.0,
        nondefault_nsrc=None,
        normalize_audio=False,
    ):
        super(WhamDataset, self).__init__()
        if task not in WHAM_TASKS.keys():
            raise ValueError(
                "Unexpected task {}, expected one of " "{}".format(task, WHAM_TASKS.keys())
            )
        # Task setting
        self.json_dir = json_dir
        self.task = task
        self.task_dict = WHAM_TASKS[task]
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.seg_len = None if segment is None else int(segment * sample_rate)
        self.EPS = 1e-8
        if not nondefault_nsrc:
            self.n_src = self.task_dict["default_nsrc"]
        else:
            assert nondefault_nsrc >= self.task_dict["default_nsrc"]
            self.n_src = nondefault_nsrc
        self.like_test = self.seg_len is None
        # Load json files
        mix_json = os.path.join(json_dir, self.task_dict["mixture"] + ".json")
        # print("mix_json",mix_json)
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in self.task_dict["sources"]
        ]
        # sources_json['./voice_data/json/tr/s1.json', './voice_data/json/tr/s2.json', './voice_data/json/tr/s3.json']
        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))
        # Filter out short utterances only when segment is specified
        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0
        if not self.like_test:
            for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
                if mix_infos[i][1] < self.seg_len:
                    drop_utt += 1
                    drop_len += mix_infos[i][1]
                    del mix_infos[i]
                    for src_inf in sources_infos:
                        del src_inf[i]

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / sample_rate / 36000, orig_len, self.seg_len
            )
        )
        self.mix = mix_infos
        # Handle the case n_src > default_nsrc
        while len(sources_infos) < self.n_src:
            sources_infos.append([None for _ in range(len(self.mix))])
        self.sources = sources_infos

    def __add__(self, wham):
        if self.n_src != wham.n_src:
            raise ValueError(
                "Only datasets having the same number of sources"
                "can be added together. Received "
                "{} and {}".format(self.n_src, wham.n_src)
            )
        if self.seg_len != wham.seg_len:
            self.seg_len = min(self.seg_len, wham.seg_len)
            print(
                "Segment length mismatched between the two Dataset"
                "passed one the smallest to the sum."
            )
        self.mix = self.mix + wham.mix
        self.sources = [a + b for a, b in zip(self.sources, wham.sources)]

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Random start
        if self.mix[idx][1] == self.seg_len or self.like_test:
            rand_start = 0
        else:
            #seg_len=segment * sample_rate
            rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len
####################################
       # Load mixture
       #  print("self.mix[idx][0]",self.mix[idx][0])
        # x, _  = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype="float32")
        x=np.loadtxt(self.mix[idx][0], dtype=float,unpack=True)
        x = np.float32(x)
        seg_len = torch.as_tensor([len(x)])
        # huatu(x,"部分混合信号"+str(idx)+"seg_len"+str(seg_len))
########## Load sources
        source_arrays = []
        # print("self.sources",self.sources)
        # print("idx",idx)
        for src in self.sources:
            # print("src[idx]",src[idx])
            if src[idx] is None:
                # Target is filled with zeros if n_src > default_nsrc
                s = np.zeros((seg_len,))
            else:
                # s, _ = sf.read(src[idx][0], start=rand_start, stop=stop, dtype="float32")
                s = np.loadtxt(src[idx][0],unpack=True, dtype=float)
                s = np.float32(s)
                # print(s,s.shape)
                # print("信息不充分吧")

            source_arrays.append(s)
            """     source_arrays  [array([0.0000000e+00, 3.1415926e-05, 6.2831852e-05, ...,
                    -9.4247778e-05, -6.2831852e-05, -3.1415926e-05], dtype=float32),
             array([0.0000000e+00, 3.1415926e-05, 6.2831852e-05, ...,
                    -9.4247778e-05, -6.2831852e-05, -3.1415926e-05], dtype=float32),
             array([0.0000000e+00, 3.1415926e-05, 6.2831852e-05, ...,
                    -9.4247778e-05, -6.2831852e-05, -3.1415926e-05], dtype=float32)]"""

        sources = torch.from_numpy(np.vstack(source_arrays))
        """ sources tensor([[0.0000e+00, 3.1416e-05, 6.2832e-05, ..., -9.4248e-05,
                 -6.2832e-05, -3.1416e-05],
                [0.0000e+00, 3.1416e-05, 6.2832e-05, ..., -9.4248e-05,
                 -6.2832e-05, -3.1416e-05],
                [0.0000e+00, 3.1416e-05, 6.2832e-05, ..., -9.4248e-05,
                 -6.2832e-05, -3.1416e-05]])"""

        mixture = torch.from_numpy(x)
        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

        """    print("mixture",mixture)
        mixture
        tensor([0.0000e+00, 3.1416e-05, 6.2832e-05, ..., -9.4248e-05,
                -6.2832e-05, -3.1416e-05])
        
        print("sources",sources)
        sources
        tensor([[0.0000e+00, 3.1416e-05, 6.2832e-05, ..., -9.4248e-05,
                 -6.2832e-05, -3.1416e-05],
                [0.0000e+00, 3.1416e-05, 6.2832e-05, ..., -9.4248e-05,
                 -6.2832e-05, -3.1416e-05],
                [0.0000e+00, 3.1416e-05, 6.2832e-05, ..., -9.4248e-05,
                 -6.2832e-05, -3.1416e-05]])"""
        return mixture, sources

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = self.task
        if self.task == "sep_clean":
            data_license = [wsj0_license]
        else:
            data_license = [wsj0_license, wham_noise_license]
        infos["licenses"] = data_license
        return infos





wham_noise_license = dict(
    title="The WSJ0 Hipster Ambient Mixtures dataset",
    title_link="http://wham.whisper.ai/",
    author="Whisper.ai",
    author_link="https://whisper.ai/",
    license="CC BY-NC 4.0",
    license_link="https://creativecommons.org/licenses/by-nc/4.0/",
    non_commercial=True,
)


####原版wham

# class WhamDataset(data.Dataset):
#     """Dataset class for WHAM source separation and speech enhancement tasks.
#
#     Args:
#         json_dir (str): The path to the directory containing the json files.
#         task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
#             ``'sep_noisy'``.
#
#             * ``'enh_single'`` for single speaker speech enhancement.
#             * ``'enh_both'`` for multi speaker speech enhancement.
#             * ``'sep_clean'`` for two-speaker clean source separation.
#             * ``'sep_noisy'`` for two-speaker noisy source separation.
#
#         sample_rate (int, optional): The sampling rate of the wav files.
#         segment (float, optional): Length of the segments used for training,
#             in seconds. If None, use full utterances (e.g. for test).
#         nondefault_nsrc (int, optional): Number of sources in the training
#             targets.
#             If None, defaults to one for enhancement tasks and two for
#             separation tasks.
#         normalize_audio (bool): If True then both sources and the mixture are
#             normalized with the standard deviation of the mixture.
#
#     References
#         "WHAM!: Extending Speech Separation to Noisy Environments",
#         Wichern et al. 2019
#     """
#
#     dataset_name = "WHAM"
#
#     def __init__(
#         self,
#         json_dir,
#         task,
#         sample_rate=8000,
#         segment=4.0,
#         nondefault_nsrc=None,
#         normalize_audio=False,
#     ):
#         super(WhamDataset, self).__init__()
#         if task not in WHAM_TASKS.keys():
#             raise ValueError(
#                 "Unexpected task {}, expected one of " "{}".format(task, WHAM_TASKS.keys())
#             )
#         # Task setting
#         self.json_dir = json_dir
#         self.task = task
#         self.task_dict = WHAM_TASKS[task]
#         self.sample_rate = sample_rate
#         self.normalize_audio = normalize_audio
#         self.seg_len = None if segment is None else int(segment * sample_rate)
#         self.EPS = 1e-8
#         if not nondefault_nsrc:
#             self.n_src = self.task_dict["default_nsrc"]
#         else:
#             assert nondefault_nsrc >= self.task_dict["default_nsrc"]
#             self.n_src = nondefault_nsrc
#         self.like_test = self.seg_len is None
#         # Load json files
#         mix_json = os.path.join(json_dir, self.task_dict["mixture"] + ".json")
#         sources_json = [
#             os.path.join(json_dir, source + ".json") for source in self.task_dict["sources"]
#         ]
#         with open(mix_json, "r") as f:
#             mix_infos = json.load(f)
#         sources_infos = []
#         for src_json in sources_json:
#             with open(src_json, "r") as f:
#                 sources_infos.append(json.load(f))
#         # Filter out short utterances only when segment is specified
#         orig_len = len(mix_infos)
#         drop_utt, drop_len = 0, 0
#         if not self.like_test:
#             for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
#                 if mix_infos[i][1] < self.seg_len:
#                     drop_utt += 1
#                     drop_len += mix_infos[i][1]
#                     del mix_infos[i]
#                     for src_inf in sources_infos:
#                         del src_inf[i]
#
#         print(
#             "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
#                 drop_utt, drop_len / sample_rate / 36000, orig_len, self.seg_len
#             )
#         )
#         self.mix = mix_infos
#         # Handle the case n_src > default_nsrc
#         while len(sources_infos) < self.n_src:
#             sources_infos.append([None for _ in range(len(self.mix))])
#         self.sources = sources_infos
#
#     def __add__(self, wham):
#         if self.n_src != wham.n_src:
#             raise ValueError(
#                 "Only datasets having the same number of sources"
#                 "can be added together. Received "
#                 "{} and {}".format(self.n_src, wham.n_src)
#             )
#         if self.seg_len != wham.seg_len:
#             self.seg_len = min(self.seg_len, wham.seg_len)
#             print(
#                 "Segment length mismatched between the two Dataset"
#                 "passed one the smallest to the sum."
#             )
#         self.mix = self.mix + wham.mix
#         self.sources = [a + b for a, b in zip(self.sources, wham.sources)]
#
#     def __len__(self):
#         return len(self.mix)
#
#     def __getitem__(self, idx):
#         """Gets a mixture/sources pair.
#         Returns:
#             mixture, vstack([source_arrays])
#         """
#         # Random start
#         if self.mix[idx][1] == self.seg_len or self.like_test:
#             rand_start = 0
#         else:
#             rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
#         if self.like_test:
#             stop = None
#         else:
#             stop = rand_start + self.seg_len
# ####################################
#        # Load mixture
#         x, _  = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype="float32")
#         seg_len = torch.as_tensor([len(x)])
# ########## Load sources
#         source_arrays = []
#         for src in self.sources:
#             if src[idx] is None:
#                 # Target is filled with zeros if n_src > default_nsrc
#                 s = np.zeros((seg_len,))
#             else:
#                 s, _ = sf.read(src[idx][0], start=rand_start, stop=stop, dtype="float32")
#             source_arrays.append(s)
#         sources = torch.from_numpy(np.vstack(source_arrays))
#         mixture = torch.from_numpy(x)
#
#         if self.normalize_audio:
#             m_std = mixture.std(-1, keepdim=True)
#             mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
#             sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)
#         return mixture, sources
#
#     def get_infos(self):
#         """Get dataset infos (for publishing models).
#
#         Returns:
#             dict, dataset infos with keys `dataset`, `task` and `licences`.
#         """
#         infos = dict()
#         infos["dataset"] = self.dataset_name
#         infos["task"] = self.task
#         if self.task == "sep_clean":
#             data_license = [wsj0_license]
#         else:
#             data_license = [wsj0_license, wham_noise_license]
#         infos["licenses"] = data_license
#         return infosyu