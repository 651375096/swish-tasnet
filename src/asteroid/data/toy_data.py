import torch
from torch.utils import data


# license = dict(
#     title='',
#     title_link='',
#     author='',
#     author_link='',
#     licence='',
#     licence_link='',
#     non_commercial=True
# )


class WavSet(data.Dataset):
    def __init__(self, n_src=2, ex_len=32000, n_ex=1000):
        self.n_src = n_src
        self.ex_len = ex_len
        self.n_ex = n_ex

    def __len__(self):
        return self.n_ex

    def __getitem__(self, idx):
        mixture = torch.randn(1, self.ex_len)
        sources = torch.randn(self.n_src, self.ex_len)
        return mixture, sources


class SpecSet(data.Dataset):
    def __init__(self, n_src=2, n_frames=600, n_freq=512, n_ex=1000):
        self.n_src = n_src
        self.n_frames = n_frames
        self.n_freq = n_freq
        self.n_ex = n_ex

    def __len__(self):
        return self.n_ex

    def __getitem__(self, idx):
        mixture = torch.randn(1, self.n_freq, self.n_frames)
        sources = torch.randn(self.n_src, self.n_freq, self.n_frames)
        return mixture, sources
