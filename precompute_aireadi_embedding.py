import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
os.environ["HF_HOME"] = "/playpen/haochenz/hf_cache"

from models.encoders.qwen3_vl_embedding import Qwen3VLEmbedder


# ==========================
# PATH CONFIG
# ==========================

DATA_ROOT = "/playpen-shared/haochenz/AI-READI"
# DATA_ROOT = "/Users/zhc/Documents/AI-READI"

RETINAL_ROOT = os.path.join(
    DATA_ROOT,
    "retinal_photography/cfp/topcon_maestro2"
)

META_FILE = os.path.join(DATA_ROOT, "participants.tsv")

SAVE_DIR = os.path.join(DATA_ROOT, "precomputed_embeddings")
os.makedirs(SAVE_DIR, exist_ok=True)

# resize as (H, W)
RETINAL_RESIZE = (256, 337)


# ==========================
# Load Qwen3VL embedding model
# ==========================

def load_model():
    model_name = "Qwen/Qwen3-VL-Embedding-2B"
    # model_name = "/Users/zhc/Downloads/Qwen3-VL-Embedding-2B"

    model = Qwen3VLEmbedder(
        model_name_or_path=model_name,
        torch_dtype=torch.float16,
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
# image utils
# ==========================

def read_and_resize_image(image_path, retinal_resize):
    """
    Read image as PIL RGB and resize to fixed size.
    retinal_resize is (H, W)
    """
    img = Image.open(image_path).convert("RGB")

    if retinal_resize is not None:
        h, w = retinal_resize
        img = img.resize((w, h))

    return img


def load_retinal_images(patient_dir, retinal_resize):
    """
    Return 4 resized PIL images in a fixed order.
    If one of them is missing, return None.
    """
    order = [
        "macula_oct_cfp_l",
        "macula_oct_cfp_r",
        "wide_oct_cfp_l",
        "wide_oct_cfp_r",
    ]

    files = os.listdir(patient_dir)
    images = []

    for key in order:
        matches = sorted([f for f in files if key in f])

        if len(matches) == 0:
            return None

        image_path = os.path.join(patient_dir, matches[0])
        img = read_and_resize_image(image_path, retinal_resize)
        images.append(img)

    return images


# ==========================
# embedding helpers
# ==========================

@torch.no_grad()
def get_text_embedding(model, text):
    emb = model.process([{"text": text}])
    return emb.cpu()


@torch.no_grad()
def get_retinal_embedding(model, images):
    """
    images: list of 4 PIL images
    return: averaged embedding over 4 retinal images
    """
    inputs = [{"image": img} for img in images]
    img_embeds = model.process(inputs)   # [4, D]
    # retinal_embed = img_embeds.mean(dim=0)
    # return retinal_embed.cpu()
    return img_embeds.cpu()

# ==========================
# Precompute embeddings
# ==========================

def run_precompute():
    model = load_model()
    meta_df = pd.read_csv(META_FILE, sep="\t")

    patient_ids = sorted(os.listdir(RETINAL_ROOT))

    text_embeddings = {}
    retinal_embeddings = {}
    valid_patient_ids = []

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

        # =================
        # retinal embedding
        # =================
        retinal_images = load_retinal_images(patient_dir, RETINAL_RESIZE)
        if retinal_images is None:
            continue

        # try:
        #     text_embed = get_text_embedding(model, text)
        #     retinal_embed = get_retinal_embedding(model, retinal_images)
        # except Exception as e:
        #     print(f"[WARNING] Failed on patient {pid}: {e}")
        #     continue

        text_embed = get_text_embedding(model, text)
        retinal_embed = get_retinal_embedding(model, retinal_images)


        text_embeddings[pid] = text_embed
        retinal_embeddings[pid] = retinal_embed
        valid_patient_ids.append(pid)

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

    torch.save(
        valid_patient_ids,
        os.path.join(SAVE_DIR, "patient_ids.pt")
    )

    print("Saved embeddings")
    print("text:", len(text_embeddings))
    print("retinal:", len(retinal_embeddings))
    print("patients:", len(valid_patient_ids))
    print("retinal resize:", RETINAL_RESIZE)


# ==========================

if __name__ == "__main__":
    run_precompute()
    # # scp -r /Users/zhc/Documents/AI-READI/participants.tsv haochenz@unites1.cs.unc.edu:/playpen/haochenz/AI-READI