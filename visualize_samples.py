import torch
import matplotlib.pyplot as plt
import numpy as np
from momentfm import MOMENTPipeline
from tqdm import tqdm


def _moment_embed(moment_model, x, device, batch_size, save_path):
    """
    x: torch.Tensor, shape (N, n_var, seq_len)
    return: np.ndarray, shape (N, dim)
    """
    moment_model.eval()
    emb_list = []

    with torch.no_grad():
        for start in tqdm(range(0, x.shape[0], batch_size)):
            batch = x[start:start + batch_size].to(device).float()
            print(f"batch shape: {batch.shape}")
            out = moment_model(x_enc=batch, reduction="none").embeddings
            # out shape: (B, n_var, seq_len, dim)
            # out = out.mean(dim=(1, 2))   # -> (B, dim)
            # breakpoint()
            emb_list.append(out.cpu().numpy())
    moment_emb = np.concatenate(emb_list, axis=0)
    np.save(save_path, moment_emb)



# samples = torch.load("./results/samples_synth_u_causal.pt")
#
# for i in range(len(samples['real_ts'])):
#     real = samples['real_ts'][i,:,:32].flatten()
#     plt.plot(real, label="real", color="orange")
#     for j in range(10):
#         fake = samples['sampled_ts'][j, i,:,:32].flatten()
#         plt.plot(fake, label="fake", color="blue", alpha=0.5)
#     # plt.legend()
#     plt.show()
#     print(i)
#     if i > 6:
#         break

if __name__ == "__main__":
    # train_ts = np.load("/Users/zhc/Documents/LitsDatasets/128_len_ts/synthetic_u/train_ts.npy")
    moment_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "embedding"},
        )
    moment_model.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moment_model = moment_model.to(device)
    moment_model.eval()


    path_list = [
        "/playpen-shared/haochenz/LitsDatasets/128_len_ts/synthetic_u",
        "/playpen-shared/haochenz/LitsDatasets/128_len_ts/synthetic_m",
    ]
    for path in path_list:
        for split in ["train", "valid", "test"]:
            print(f"processing {path}")
            _ts = np.load(f"{path}/{split}_ts.npy")
            _ts = torch.from_numpy(_ts).float().to(device).permute(0, 2, 1)
            _moment_embed(
                moment_model, _ts, device, batch_size=64,
                save_path=f"{path}/{split}_moment_embeds.npy"
            )
