import os
import json
import numpy as np
import random
from torch.utils.data import Dataset
import time
import torch

class CustomDataset:
    def __init__(self, folder, **kwargs):
        super().__init__()
        self.folder = folder
        self._load_meta()

    def _load_meta(self):
        self.meta = json.load(open(os.path.join(self.folder, "meta.json")))
        self.attr_list = self.meta["attr_list"]
        n_attr = len(self.attr_list)
        self.attr_ids = np.arange(n_attr)
        self.attr_n_ops = np.array(self.meta["attr_n_ops"])

    def get_split(self, split, text_type,  *args):
        return CustomSplit(self.folder, text_type, split)


class CustomSplit(Dataset):
    def __init__(self, folder, text_type, split="train"):
        super().__init__()
        assert split in ("train", "valid", "test"), "Please specify a valid split."
        self.split = split            
        self.folder = folder
        self.text_type = text_type
        self._load_data()

        print(f"Split: {self.split}, total samples {self.n_samples}.")

    def _load_data(self):
        ts = np.load(os.path.join(self.folder, self.split+"_ts.npy"))     # [n_samples, n_steps]
        attrs = np.load(os.path.join(self.folder, self.split+"_attrs_idx.npy"))  # [n_samples, n_attrs]

        if self.text_type == "original_text":
            caps = np.load(os.path.join(self.folder, self.split+fr"_text_caps.npy"), allow_pickle=True) # need to change if I want
        elif self.text_type == "my_generated_text":
            caps = np.load(os.path.join(self.folder, self.split+fr"_text_my_caps.npy"), allow_pickle=True) # need to change if I want
        else:
            raise NotImplementedError

        if self.text_type == "original_text":
            caps_embed_path = os.path.join(self.folder, self.split+fr"_embeds_caps.pt")
            if os.path.exists(caps_embed_path):
                # raise FileNotFoundError(f"Embedding file not found: {caps_embed_path}")
                # caps_embed = np.load(caps_embed_path, allow_pickle=True)
                caps_embed = torch.load(caps_embed_path, map_location="cpu")
                self.caps_embed = caps_embed
                print("using precomputed caps embedding.")
            else:
                self.caps_embed = None

        self.ts, self.attrs, self.caps = ts, attrs, caps
        self.n_samples = self.ts.shape[0]
        self.n_steps = self.ts.shape[1]
        self.n_attrs = self.attrs.shape[1]
        self.time_point = np.arange(self.n_steps)

    def __getitem__(self, idx):
        cap_id = random.randint(0, len(self.caps[idx])-1)
        tmp_ts = self.ts[idx]
        if len(tmp_ts.shape) == 1:
            tmp_ts = tmp_ts[...,np.newaxis]

        item_dict = {
            "ts": tmp_ts,
            "ts_len": tmp_ts.shape[0],
            "attrs": self.attrs[idx],
            "cap": self.caps[idx][cap_id],
            "tp": self.time_point
        }

        if self.caps_embed is not None:
            item_dict.update({"cap_embed": self.caps_embed[idx]})
        return item_dict

    def __len__(self):
        return self.n_samples