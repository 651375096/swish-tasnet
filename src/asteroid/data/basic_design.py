from torch.utils.data import Dataset
from typing import Dict


class BaseDataset(Dataset):
    def __init__(self, dic: Dict):
        ...

    def from_csv(self, file):
        ...

    def from_json(self, file):
        ...

    def __getitem__(self, item):
        self.user_hook()

    def user_hook(
        self,
    ):
        ...
