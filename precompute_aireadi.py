import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

from models.encoders.qwen3_vl_embedding import Qwen3VLEmbedder


# ==========================
# PATH CONFIG
# ==========================

DATA_ROOT = "/playpen/haochenz/AI-READI"

RETINAL_ROOT = os.path.join(
    DATA_ROOT,
    "retinal_photography/cfp/topcon_maestro2"
)

META_FILE = os.path.join(DATA_ROOT, "participants.tsv")

SAVE_DIR = os.path.join(DATA_ROOT, "precomputed_embeddings")

os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================
# Load Qwen3VL embedding model
# ==========================

def load_model():

    model_name = "Qwen/Qwen3-VL-Embedding-2B"

    model = Qwen3VLEmbedder(
        model_name_or_path=model_name,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )

    return model


# ==========================
# build text description
# ==========================

def build_text_description(meta_df, patient_id):

    row = meta_df.query(f"person_id == {patient_id}")

    if len(row) == 0:
        return None

    age = row["age"].item()
    study_group = row["study_group"].item()

    text = f"This patient is {age} years old, their status is {study_group}"

    return text


# ==========================
# read retinal images
# ==========================

def load_retinal_images(patient_dir):

    order = [
        "macula_oct_cfp_l",
        "macula_oct_cfp_r",
        "wide_oct_cfp_l",
        "wide_oct_cfp_r",
    ]

    files = os.listdir(patient_dir)

    images = []

    for key in order:

        matches = [f for f in files if key in f]

        if len(matches) == 0:
            return None

        path = os.path.join(patient_dir, matches[0])

        images.append(path)

    return images


# ==========================
# Precompute embeddings
# ==========================

def run_precompute():

    model = load_model()

    meta_df = pd.read_csv(META_FILE, sep="\t")

    patient_ids = os.listdir(RETINAL_ROOT)

    text_embeddings = {}
    retinal_embeddings = {}

    for pid in tqdm(patient_ids):

        patient_dir = os.path.join(RETINAL_ROOT, pid)

        if not os.path.isdir(patient_dir):
            continue

        # =================
        # text embedding
        # =================

        text = build_text_description(meta_df, int(pid))

        if text is None:
            continue

        text_embed = model.process([{"text": text}])

        text_embeddings[pid] = text_embed.squeeze().cpu()

        # =================
        # retinal embedding
        # =================

        image_paths = load_retinal_images(patient_dir)

        if image_paths is None:
            continue

        inputs = []

        for img_path in image_paths:
            inputs.append({"image": img_path})

        img_embeds = model.process(inputs)

        # average 4 retinal embeddings
        retinal_embed = img_embeds.mean(0)

        retinal_embeddings[pid] = retinal_embed.cpu()

    # =================
    # save
    # =================

    torch.save(
        text_embeddings,
        os.path.join(SAVE_DIR, "text_embeddings.pt")
    )

    torch.save(
        retinal_embeddings,
        os.path.join(SAVE_DIR, "retinal_embeddings.pt")
    )

    print("Saved embeddings")
    print("text:", len(text_embeddings))
    print("retinal:", len(retinal_embeddings))


# ==========================

if __name__ == "__main__":
    run_precompute()