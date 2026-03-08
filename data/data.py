# class CustomDataset:
#     def __init__(self, folder, **kwargs):
#         super().__init__()
#         self.folder = folder
#         # self._load_meta()
#
#     # def _load_meta(self):
#     #     self.meta = json.load(open(os.path.join(self.folder, "meta.json")))
#     #     self.attr_list = self.meta["attr_list"]
#     #     n_attr = len(self.attr_list)
#     #     self.attr_ids = np.arange(n_attr)
#     #     self.attr_n_ops = np.array(self.meta["attr_n_ops"])
#
#     def get_split(self, split, text_type,  *args):
#         return CustomSplit(self.folder, text_type, split)
#
#
# class CustomSplit(Dataset):
#     def __init__(self, folder, text_type, split="train"):
#         super().__init__()
#         assert split in ("train", "valid", "test"), "Please specify a valid split."
#         self.split = split
#         self.folder = folder
#         self.text_type = text_type
#         self._load_data()
#
#         print(f"Split: {self.split}, total samples {self.n_samples}.")
#
#     def _load_data(self):
#         ts = np.load(os.path.join(self.folder, self.split+"_ts.npy"))     # [n_samples, n_steps]
#         attrs = np.load(os.path.join(self.folder, self.split+"_attrs_idx.npy"))  # [n_samples, n_attrs]
#
#         if "original_text" in self.text_type:
#             caps = np.load(os.path.join(self.folder, self.split+fr"_text_caps.npy"), allow_pickle=True) # need to change if I want
#         elif "my_generated_text" in self.text_type:
#             caps = np.load(os.path.join(self.folder, self.split+fr"_text_my_caps_v2.npy"), allow_pickle=True) # need to change if I want
#         else:
#             raise NotImplementedError
#
#         self.caps_embed = None
#
#         if self.text_type == "original_text_embeds":
#             caps_embed_path = os.path.join(self.folder, self.split+fr"_embeds_caps.pt")
#             if os.path.exists(caps_embed_path):
#                 # raise FileNotFoundError(f"Embedding file not found: {caps_embed_path}")
#                 # caps_embed = np.load(caps_embed_path, allow_pickle=True)
#                 caps_embed = torch.load(caps_embed_path, map_location="cpu")
#                 self.caps_embed = caps_embed
#                 print("using precomputed caps embedding.")
#         elif self.text_type == "my_generated_text_embeds":
#             # caps_embed_path = os.path.join(self.folder, self.split + fr"_embeds_my_caps.pt")
#             caps_embed_path = os.path.join(self.folder, self.split + fr"_text_my_caps_v2_embeds.pt")
#             if os.path.exists(caps_embed_path):
#                 # raise FileNotFoundError(f"Embedding file not found: {caps_embed_path}")
#                 # caps_embed = np.load(caps_embed_path, allow_pickle=True)
#                 caps_embed = torch.load(caps_embed_path, map_location="cpu")
#                 self.caps_embed = caps_embed
#                 print("using precomputed caps embedding.")
#
#
#         self.ts, self.attrs, self.caps = ts, attrs, caps
#         self.n_samples = self.ts.shape[0]
#         self.n_steps = self.ts.shape[1]
#         self.n_attrs = self.attrs.shape[1]
#         self.time_point = np.arange(self.n_steps)
#
#     def __getitem__(self, idx):
#         cap_id = random.randint(0, len(self.caps[idx])-1)
#         tmp_ts = self.ts[idx]
#         if len(tmp_ts.shape) == 1:
#             tmp_ts = tmp_ts[...,np.newaxis]
#
#         item_dict = {
#             "ts": tmp_ts,
#             "ts_len": tmp_ts.shape[0],
#             "attrs": self.attrs[idx],
#             "cap": self.caps[idx][cap_id],
#             "tp": self.time_point
#         }
#
#         if self.caps_embed is not None:
#             item_dict.update({"cap_embed": self.caps_embed[idx]})
#         return item_dict
#
#     def __len__(self):
#         return self.n_samples
#







####################
####################
####################
####################
####################
####################


import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt


def _parse_patient_id(x):
    """
    Convert patient_id field from parquet into a clean string id like '1001'.
    Supports ndarray / list / tuple / str / empty values.
    """
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        x = x[0]
    elif isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        x = x[0]
    elif x is None:
        return None

    x = str(x).strip()
    if len(x) == 0 or x.lower() == "nan":
        return None

    if x.startswith("AIREADI-"):
        x = x.replace("AIREADI-", "")

    return x


def aireadi_collate_fn(batch):
    """
    Custom collate function for AI-READI.
    Returns:
        glucose_window: (B, T) float32
        retinal_images: (B, 4, 3, H, W) float32 in [0,1]
        age: (B,) long
        patient_id: list[str or int]
        study_group: list[str]
        text_description: list[str]
        time_local: list
    """
    out = {}

    out["glucose_window"] = torch.as_tensor(
        np.stack([b["glucose_window"] for b in batch], axis=0),
        dtype=torch.float32,
    )


    if "retinal_images" in batch[0]:
        # input: (B, 4, H, W, 3) uint8
        retinal = np.stack([b["retinal_images"] for b in batch], axis=0)
        retinal = torch.from_numpy(retinal).permute(0, 1, 4, 2, 3).float() / 255.0
        out["retinal_images"] = retinal

    out["age"] = torch.as_tensor(
        [b["age"] for b in batch],
        dtype=torch.long,
    )

    out["patient_id"] = [b["patient_id"] for b in batch]
    out["study_group"] = [b["study_group"] for b in batch]
    out["text_description"] = [b["text_description"] for b in batch]
    out["time_local"] = [b["time_local"] for b in batch]

    return out


class CustomDataset:
    def __init__(self, folder, **kwargs):
        super().__init__()
        self.folder = folder
        self.attr_n_ops = None

    def get_split(self, split, text_type, *args):
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
        ts = np.load(os.path.join(self.folder, self.split + "_ts.npy"))
        attrs = np.load(os.path.join(self.folder, self.split + "_attrs_idx.npy"))

        if "original_text" in self.text_type:
            caps = np.load(
                os.path.join(self.folder, self.split + "_text_caps.npy"),
                allow_pickle=True,
            )
        elif "my_generated_text" in self.text_type:
            caps = np.load(
                os.path.join(self.folder, self.split + "_text_my_caps_v2.npy"),
                allow_pickle=True,
            )
        else:
            raise NotImplementedError

        self.caps_embed = None

        if self.text_type == "original_text_embeds":
            caps_embed_path = os.path.join(self.folder, self.split + "_embeds_caps.pt")
            if os.path.exists(caps_embed_path):
                caps_embed = torch.load(caps_embed_path, map_location="cpu")
                self.caps_embed = caps_embed
                print("using precomputed caps embedding.")
        elif self.text_type == "my_generated_text_embeds":
            caps_embed_path = os.path.join(
                self.folder, self.split + "_text_my_caps_v2_embeds.pt"
            )
            if os.path.exists(caps_embed_path):
                caps_embed = torch.load(caps_embed_path, map_location="cpu")
                self.caps_embed = caps_embed
                print("using precomputed caps embedding.")

        self.ts, self.attrs, self.caps = ts, attrs, caps
        self.n_samples = self.ts.shape[0]
        self.n_steps = self.ts.shape[1]
        self.n_attrs = self.attrs.shape[1]
        self.time_point = np.arange(self.n_steps)

    def __getitem__(self, idx):
        cap_id = random.randint(0, len(self.caps[idx]) - 1)
        tmp_ts = self.ts[idx]
        if len(tmp_ts.shape) == 1:
            tmp_ts = tmp_ts[..., np.newaxis]

        item_dict = {
            "ts": tmp_ts,
            "ts_len": tmp_ts.shape[0],
            "attrs": self.attrs[idx],
            "cap": self.caps[idx][cap_id],
            "tp": self.time_point,
        }

        if self.caps_embed is not None:
            item_dict.update({"cap_embed": self.caps_embed[idx]})
        return item_dict

    def __len__(self):
        return self.n_samples


class AIREADIDataset:
    """
    Wrapper class to make AI-READI compatible with the existing
    GenerationDataset -> get_split(...) design.
    """

    def __init__(
        self,
        folder=None,
        data_path=None,
        window_size=24,
        retinal_resize=(256, 256),
        metadata_path=None,
        retinal_root=None,
        glucose_prefix="glucose",
        glucose_min=40.0,
        glucose_max=400.0,
        **kwargs,
    ):
        self.data_path = data_path if data_path is not None else folder
        if self.data_path is None:
            raise ValueError("AIREADIDataset requires either `data_path` or `folder`.")

        self.window_size = int(window_size)
        self.retinal_resize = retinal_resize
        self.metadata_path = (
            metadata_path
            if metadata_path is not None
            else os.path.join(self.data_path, "participants.tsv")
        )
        self.retinal_root = (
            retinal_root
            if retinal_root is not None
            else os.path.join(self.data_path, "retinal_photography", "cfp", "topcon_maestro2")
        )
        self.glucose_prefix = glucose_prefix
        self.glucose_min = float(glucose_min)
        self.glucose_max = float(glucose_max)

        self.attr_n_ops = None

    def get_split(self, split, text_type=None, *args):
        return AIREADISplit(
            data_path=self.data_path,
            split=split,
            window_size=self.window_size,
            retinal_resize=self.retinal_resize,
            metadata_path=self.metadata_path,
            retinal_root=self.retinal_root,
            glucose_prefix=self.glucose_prefix,
            glucose_min=self.glucose_min,
            glucose_max=self.glucose_max,
        )


class AIREADISplit(Dataset):
    def __init__(
        self,
        data_path,
        split="train",
        window_size=24,
        retinal_resize=(256, 256),
        metadata_path=None,
        retinal_root=None,
        glucose_prefix="glucose",
        glucose_min=40.0,
        glucose_max=400.0,
    ):
        super().__init__()
        assert split in ("train", "valid", "test"), "Please specify a valid split."

        self.data_path = data_path
        self.split = split
        self.window_size = int(window_size)
        self.retinal_resize = retinal_resize
        self.metadata_path = metadata_path
        self.retinal_root = retinal_root
        self.glucose_prefix = glucose_prefix
        self.glucose_min_ = float(glucose_min)
        self.glucose_max_ = float(glucose_max)

        self.collate_fn = aireadi_collate_fn

        self.load_meta_data()
        self.load_glucose()
        self.cache_retinal_images()
        self.build_windows()

        print(f"Split: {self.split}, total samples {len(self.windows)}.")

    def load_meta_data(self):
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        meta = pd.read_csv(self.metadata_path, sep="\t")
        meta["person_id"] = meta["person_id"].astype(str)
        self.meta_df = meta
        self.meta_dict = meta.set_index("person_id").to_dict("index")

    def load_glucose(self):
        parquet_path = os.path.join(self.data_path, f"{self.glucose_prefix}_{self.split}.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        if not os.path.isdir(self.retinal_root):
            raise FileNotFoundError(f"Retinal root not found: {self.retinal_root}")

        df = pd.read_parquet(parquet_path).copy()

        if "patient_id" not in df.columns:
            raise KeyError("Expected column `patient_id` in parquet.")

        df["patient_id"] = df["patient_id"].apply(_parse_patient_id)
        df = df.dropna(subset=["patient_id"])

        valid_ids = {
            f for f in os.listdir(self.retinal_root)
            if os.path.isdir(os.path.join(self.retinal_root, f))
        }
        df = df[df["patient_id"].isin(valid_ids)]

        # Keep only patients that also exist in participants.tsv
        df = df[df["patient_id"].isin(set(self.meta_dict.keys()))]

        df = df.sort_values(["patient_id"]).reset_index(drop=True)

        print(f"[AI-READI:{self.split}] rows: {len(df)}")
        print(f"[AI-READI:{self.split}] patients: {df['patient_id'].nunique()}")

        self.glucose_df = df
        self.patient_groups = dict(tuple(df.groupby("patient_id", sort=False)))

    def _read_one_retinal_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.retinal_resize is not None:
            # retinal_resize = (H, W) or (width, height)? We will interpret as (H, W)
            h, w = self.retinal_resize
            img = img.resize((w, h))

        img_np = np.array(img, dtype=np.uint8)
        # plt.imshow(img_np)
        # plt.title(path)
        # plt.axis("off")
        # plt.show()
        return img_np

    def cache_retinal_images(self):
        print(f"[AI-READI:{self.split}] Caching retinal images...")

        self.retinal_cache = {}
        self.retinal_shape = None

        order = [
            "macula_oct_cfp_l",
            "macula_oct_cfp_r",
            "wide_oct_cfp_l",
            "wide_oct_cfp_r",
        ]

        for pid in self.patient_groups.keys():
            patient_path = os.path.join(self.retinal_root, pid)
            if not os.path.isdir(patient_path):
                continue

            files = os.listdir(patient_path)
            imgs = []

            ok = True
            for key in order:
                matches = sorted([f for f in files if key in f])
                if len(matches) == 0:
                    ok = False
                    break

                path = os.path.join(patient_path, matches[0])
                img = self._read_one_retinal_image(path)
                imgs.append(img)

            if not ok or len(imgs) != 4:
                continue

            # Check within-patient consistency
            shapes = {img.shape for img in imgs}
            if len(shapes) != 1:
                continue

            stacked = np.stack(imgs, axis=0)  # (4, H, W, 3)

            # Enforce global consistency if no resize is applied
            if self.retinal_shape is None:
                self.retinal_shape = stacked.shape
            else:
                if stacked.shape != self.retinal_shape:
                    continue

            self.retinal_cache[pid] = stacked

        print(f"[AI-READI:{self.split}] retinal cached: {len(self.retinal_cache)}")
        if self.retinal_shape is not None:
            print(f"[AI-READI:{self.split}] retinal shape: {self.retinal_shape}")

    def _extract_patient_sequence(self, g):
        """
        Robustly extract glucose and time sequence for one patient group.
        This supports both:
        - one row per patient, where row['glucose'] is a full array
        - multiple rows per patient, each row containing a segment
        """
        glucose_parts = []
        time_parts = []

        for _, row in g.iterrows():
            values = np.asarray(row["glucose"])
            times = np.asarray(row["time_local"])

            if values.ndim == 0:
                values = np.asarray([values], dtype=np.float32)
            values = values.astype(np.float32)

            if times.ndim == 0:
                times = np.asarray([times])

            n = min(len(values), len(times))
            if n <= 0:
                continue

            glucose_parts.append(values[:n])
            time_parts.append(times[:n])

        if len(glucose_parts) == 0:
            return None, None

        values = np.concatenate(glucose_parts, axis=0)
        times = np.concatenate(time_parts, axis=0)

        return values, times

    def build_windows(self):
        self.windows = []
        self.patient_series = {}

        for pid, g in self.patient_groups.items():
            if pid not in self.retinal_cache:
                continue
            if pid not in self.meta_dict:
                continue

            values, times = self._extract_patient_sequence(g)
            if values is None:
                continue

            self.patient_series[pid] = {
                "glucose": values,
                "time_local": times,
            }


            n_windows = len(values) - self.window_size + 1

            if n_windows <= 0:
                continue

            for start_idx in range(n_windows):
                self.windows.append((pid, start_idx))

        print(f"[AI-READI:{self.split}] total windows: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        pid, start_idx = self.windows[idx]

        series = self.patient_series[pid]
        values = series["glucose"]
        times = series["time_local"]

        window = values[start_idx:start_idx + self.window_size]
        window = (window - self.glucose_min_) / (self.glucose_max_ - self.glucose_min_)
        window = window.astype(np.float32)


        time_local = times[start_idx:start_idx + self.window_size]

        meta = self.meta_dict[pid]
        age = int(meta["age"]) if not pd.isna(meta["age"]) else -1
        study_group = str(meta["study_group"]) if not pd.isna(meta["study_group"]) else "unknown"

        sample = {
            "glucose_window": window,
            "time_local": time_local,
            # "patient_id": int(pid) if str(pid).isdigit() else pid,
            # "age": age,
            # "study_group": study_group,
            "text_description": f"This patient is {age} years old, their status is {study_group}",
            "retinal_images": self.retinal_cache[pid],
        }



        return sample


