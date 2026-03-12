import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os

os.environ["HF_HOME"] = "/playpen/haochenz/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/playpen/haochenz/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/playpen/haochenz/hf_cache"

def embed_numpy_to_pt(
    texts_array: np.ndarray,          # numpy array of strings
    output_pt_path: str,
    batch_size: int = 32,
    device: str = "cuda"
):
    """
    texts_array: np.ndarray of shape (N,) containing strings
    """

    assert isinstance(texts_array, np.ndarray)
    assert texts_array.dtype.type is np.str_ or texts_array.dtype == object

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    model.eval().to(device)

    all_embeddings = []

    N = len(texts_array)

    for start in tqdm(range(0, N, batch_size), desc="Encoding"):
        if start == 0:
            print(texts_array[start][0])
        end = min(start + batch_size, N)
        # batch_texts = texts_array[start:end].tolist()
        batch_texts = [t[0] for t in texts_array[start:end]]
        emb = model.encode(
            batch_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=device,
        )

        all_embeddings.append(emb.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)

    torch.save(embeddings, output_pt_path)

    print(f"Saved embeddings to {output_pt_path}")
    print("embeddings shape:", embeddings.shape)


if __name__ == "__main__":
    embed_numpy_to_pt(
        texts_array=np.load("/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/train_text_my_caps_v2.npy", allow_pickle=True),
        output_pt_path="/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/train_text_my_caps_v2_embeds_qwen06b.pt",
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    embed_numpy_to_pt(
        texts_array=np.load("/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/test_text_my_caps_v2.npy", allow_pickle=True),
        output_pt_path="/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/test_text_my_caps_v2_embeds_qwen06b.pt",
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    embed_numpy_to_pt(
        texts_array=np.load("/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/valid_text_my_caps_v2.npy", allow_pickle=True),
        output_pt_path="/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/valid_text_my_caps_v2_embeds_qwen06b.pt",
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # embed_numpy_to_pt(
    #     texts_array=np.load("/playpen/haochenz/synthetic_u/DiTDH-S-samples_v2.npy", allow_pickle=True),
    #     output_pt_path="/playpen/haochenz/synthetic_u/DiTDH-S-samples_v2_embeds_qwen06b.pt",
    #     batch_size=64,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )
    #
    #
    # embed_numpy_to_pt(
    #     texts_array=np.load("/playpen/haochenz/synthetic_u/train_text_my_caps.npy", allow_pickle=True),
    #     output_pt_path="/playpen/haochenz/synthetic_u/train_text_my_caps_embeds_qwen06b.pt",
    #     batch_size=64,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )
    #
    # embed_numpy_to_pt(
    #     texts_array=np.load("/playpen/haochenz/synthetic_u/test_text_my_caps.npy", allow_pickle=True),
    #     output_pt_path="/playpen/haochenz/synthetic_u/test_text_my_caps_embeds_qwen06b.pt",
    #     batch_size=64,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )
    #
    # embed_numpy_to_pt(
    #     texts_array=np.load("/playpen/haochenz/synthetic_u/valid_text_my_caps.npy", allow_pickle=True),
    #     output_pt_path="/playpen/haochenz/synthetic_u/valid_text_my_caps_embeds_qwen06b.pt",
    #     batch_size=64,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )
    #
    #
    # embed_numpy_to_pt(
    #     texts_array=np.load("/playpen/haochenz/synthetic_u/DiTDH-S-samples.npy", allow_pickle=True),
    #     output_pt_path="/playpen/haochenz/synthetic_u/DiTDH-S-samples-embeds_qwen06b.pt",
    #     batch_size=64,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )
    #
    #
    #
