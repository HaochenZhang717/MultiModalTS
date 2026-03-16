import torch
import matplotlib.pyplot as plt
import numpy as np
from momentfm import MOMENTPipeline



def _moment_embed(moment_model, x, device, batch_size=64):
    """
    x: torch.Tensor, shape (N, n_var, seq_len)
    return: np.ndarray, shape (N, dim)
    """
    moment_model.eval()
    emb_list = []

    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = x[start:start + batch_size].to(device).float()

            out = moment_model(x_enc=batch, reduction="none").embeddings
            # out shape: (B, n_var, seq_len, dim)
            out = out.mean(dim=(1, 2))   # -> (B, dim)
            breakpoint()
            emb_list.append(out.cpu().numpy())

    return np.concatenate(emb_list, axis=0)


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
    train_ts = np.load("/playpen-shared/haochenz/LitsDatasets/128_len_ts/synthetic_u/train_ts.npy")
    moment_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "embedding"},
        )
    moment_model.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ts = torch.from_numpy(train_ts).float().to(device)
    moment_model = moment_model.to(device)
    moment_model.eval()
    train_embed = _moment_embed(moment_model, train_ts, device)
