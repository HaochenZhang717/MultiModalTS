import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

os.environ["HF_HOME"] = "/playpen/haochenz/hf_cache"

from models.encoders.qwen3_vl_embedding import Qwen3VLEmbedder


# ==========================
# PATH CONFIG
# ==========================

DATA_ROOT = "/playpen-shared/haochenz/AI-READI"

RETINAL_ROOT = os.path.join(
    DATA_ROOT,
    "retinal_photography/cfp/topcon_maestro2"
)

META_FILE = os.path.join(DATA_ROOT, "participants.tsv")

SAVE_DIR = DATA_ROOT

RETINAL_RESIZE = (256, 337)


# ==========================
# Load model
# ==========================

def load_model():

    model_name = "Qwen/Qwen3-VL-Embedding-2B"

    model = Qwen3VLEmbedder(
        model_name_or_path=model_name,
        torch_dtype=torch.float16,
    )

    model.eval()

    return model


# ==========================
# Build patient text
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
# Image utils
# ==========================

def read_and_resize_image(image_path, retinal_resize):

    img = Image.open(image_path).convert("RGB")

    if retinal_resize is not None:
        h, w = retinal_resize
        img = img.resize((w, h))

    return img


def load_retinal_images(patient_dir, retinal_resize):

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
# Multimodal embedding
# ==========================

@torch.no_grad()
def get_patient_embedding(model, text, images):
    """
    text: str
    images: list of PIL images (length=4)

    return:
        patient embedding tensor [D]
    """

    inputs = []

    # text first
    inputs.append({"text": text})

    # then all images
    descriptions = [
        "left macula retinal photograph",
        "right macula retinal photograph",
        "left wide-field retinal photograph",
        "right wide-field retinal photograph",
    ]

    for img, desc in zip(images, descriptions):
        inputs.append({
            "text": desc,
            "image": img
        })



    embeds = model.process(inputs)   # shape [num_modalities, D]
    print(embeds.shape)
    breakpoint()
    # simple aggregation
    patient_embed = embeds.mean(dim=0)

    return patient_embed.cpu()


# ==========================
# Precompute embeddings
# ==========================

def run_precompute():

    model = load_model()

    meta_df = pd.read_csv(META_FILE, sep="\t")

    patient_ids = sorted(os.listdir(RETINAL_ROOT))

    patient_embeddings = {}
    valid_patient_ids = []

    for pid in tqdm(patient_ids):

        patient_dir = os.path.join(RETINAL_ROOT, pid)

        if not os.path.isdir(patient_dir):
            continue

        # =================
        # build text
        # =================

        try:
            text = build_text_description(meta_df, int(pid))
        except:
            continue

        if text is None:
            continue

        # =================
        # load retinal images
        # =================

        retinal_images = load_retinal_images(patient_dir, RETINAL_RESIZE)

        if retinal_images is None:
            continue

        # =================
        # multimodal embedding
        # =================

        try:

            patient_embed = get_patient_embedding(
                model,
                text,
                retinal_images
            )

        except Exception as e:

            print(f"[WARNING] Failed on patient {pid}: {e}")
            continue

        patient_embeddings[pid] = patient_embed

        valid_patient_ids.append(pid)

    # =================
    # save
    # =================

    torch.save(
        patient_embeddings,
        os.path.join(SAVE_DIR, "patient_embeddings.pt")
    )

    torch.save(
        valid_patient_ids,
        os.path.join(SAVE_DIR, "patient_ids.pt")
    )

    print("Saved embeddings")
    print("patients:", len(patient_embeddings))
    print("retinal resize:", RETINAL_RESIZE)


# ==========================

if __name__ == "__main__":
    run_precompute()