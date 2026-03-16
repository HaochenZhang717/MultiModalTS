import torch
import matplotlib.pyplot as plt
import numpy as np
from metrics.discriminative_torch import discriminative_score_metrics, moment_discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
import torch
import numpy as np
from scipy.linalg import sqrtm
from momentfm import MOMENTPipeline


# def visualize_sample(pt_path, idx=0):
#     data = torch.load(pt_path, map_location="cpu")
#
#     sample_dict = data[idx]
#     caption = sample_dict["caption"]
#     ts = sample_dict["sample"]  # Tensor
#     real = sample_dict["orig"]
#
#     if isinstance(ts, torch.Tensor):
#         ts = ts.numpy()
#
#     print("Caption:")
#     print(caption)
#     print("Shape:", ts.shape)
#     print("========"*30)
#
#     plt.figure(figsize=(10, 4))
#
#     # 如果是单通道
#     if ts.ndim == 1:
#         plt.plot(ts)
#     elif ts.ndim == 2:
#         # 多通道，每个通道画一条
#         for c in range(ts.shape[0]):
#             plt.plot(ts[c], label=f"Sample Channel {c}")
#
#     else:
#         raise ValueError("Unsupported time series shape")
#
#
#     if real.ndim == 1:
#         plt.plot(real)
#     elif real.ndim == 2:
#         # 多通道，每个通道画一条
#         for c in range(real.shape[0]):
#             plt.plot(real[c], label=f"real Channel {c}")
#
#     else:
#         raise ValueError("Unsupported time series shape")
#
#     plt.legend()
#     plt.title("Generated Time Series")
#     plt.xlabel("Time")
#     plt.ylabel("Value")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
#
# def calculate_scores_from_real_language(pt_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data = torch.load(pt_path, map_location=device)
#     real = []
#     fake = []
#     for datum in data:
#         real.append(datum["orig"])
#         fake.append(datum["sample"])
#     real = torch.stack(real)
#     fake = torch.stack(fake)
#
#     real = real.permute(0, 2, 1)
#     fake = fake.permute(0, 2, 1)
#     print("Real shape:", real.shape)
#     print("Fake shape:", fake.shape)
#     discriminative_score = discriminative_score_metrics(
#         real, fake,
#         real.shape[-1],
#         device,
#     )
#     print(f"Discriminative Score Metrics: {discriminative_score}")
#
#     predictive_score = predictive_score_metrics(
#         real, fake, device
#     )
#     print(f"Predictive Score Metrics: {predictive_score}")
#
#     return (
#         discriminative_score,
#         predictive_score,
#     )
#
#
# def calculate_scores_from_prior_language(sample_pt_path, real_pt_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     sample_data = torch.load(sample_pt_path, map_location=device)
#     fake = []
#     for sample_datum in sample_data:
#         fake.append(sample_datum["sample"])
#     fake = torch.stack(fake)
#
#     real_data = torch.load(real_pt_path, map_location=device)
#     real = []
#     for real_datum in real_data:
#         real.append(real_datum["orig"])
#     real = torch.stack(real)
#
#
#     real = real.permute(0, 2, 1)
#     fake = fake.permute(0, 2, 1)
#     print("Real shape:", real.shape)
#     print("Fake shape:", fake.shape)
#     discriminative_score = discriminative_score_metrics(
#         real, fake,
#         real.shape[-1],
#         device,
#     )
#     print(f"Discriminative Score Metrics: {discriminative_score}")
#
#     predictive_score = predictive_score_metrics(
#         real, fake, device
#     )
#     print(f"Predictive Score Metrics: {predictive_score}")
#
#     return (
#         discriminative_score,
#         predictive_score,
#     )
#
#
# def calculate_scores_unconditional(sample_pt_path, real_pt_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     fake = torch.load(sample_pt_path, map_location=device)
#
#     real_data = torch.load(real_pt_path, map_location=device)
#     real = []
#     for real_datum in real_data:
#         real.append(real_datum["orig"])
#     real = torch.stack(real)
#
#
#     real = real.permute(0, 2, 1)
#     fake = fake.permute(0, 2, 1)
#     print("Real shape:", real.shape)
#     print("Fake shape:", fake.shape)
#     discriminative_score = discriminative_score_metrics(
#         real, fake,
#         real.shape[-1],
#         device,
#     )
#     print(f"Discriminative Score Metrics: {discriminative_score}")
#
#     predictive_score = predictive_score_metrics(
#         real, fake, device
#     )
#     print(f"Predictive Score Metrics: {predictive_score}")
#
#     return (
#         discriminative_score,
#         predictive_score,
#     )
#
#
# def analyze_text2ts_results():
#     disc_score_list = []
#     pred_score_list = []
#     for _ in range(20):
#         # disc_score, pred_score = calculate_scores_from_prior_language(
#         #     sample_pt_path="/playpen/haochenz/ckpts_ts_generation/DiTDH-S/text2ts_from_language_prior_results.pt",
#         #     real_pt_path="/playpen/haochenz/ckpts_ts_generation/DiTDH-S/text2ts_results.pt"
#         # )
#         disc_score, pred_score = calculate_scores_from_real_language(
#             pt_path="/playpen/haochenz/ckpts_ts_generation/DiTDH-S/text2ts_results.pt")
#
#         disc_score_list.append(disc_score)
#         pred_score_list.append(pred_score)
#
#     disc_score_arr = np.array(disc_score_list)
#     pred_score_arr = np.array(pred_score_list)
#
#     disc_mean = disc_score_arr.mean()
#     disc_std = disc_score_arr.std(ddof=1)  # sample std
#
#     pred_mean = pred_score_arr.mean()
#     pred_std = pred_score_arr.std(ddof=1)
#
#     print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
#     print(f"Pred Score: mean = {pred_mean:.4f}, std = {pred_std:.4f}")


# def analyze_text2ts_results_from_prior_language():
#     disc_score_list = []
#     pred_score_list = []
#     for _ in range(20):
#         disc_score, pred_score = calculate_scores_from_prior_language(
#             sample_pt_path="/playpen/haochenz/ckpts_ts_generation/DiTDH-S/text2ts_from_language_prior_results.pt",
#             real_pt_path="/playpen/haochenz/ckpts_ts_generation/DiTDH-S/text2ts_results.pt"
#         )
#
#         disc_score_list.append(disc_score)
#         pred_score_list.append(pred_score)
#
#     disc_score_arr = np.array(disc_score_list)
#     pred_score_arr = np.array(pred_score_list)
#
#     disc_mean = disc_score_arr.mean()
#     disc_std = disc_score_arr.std(ddof=1)  # sample std
#
#     pred_mean = pred_score_arr.mean()
#     pred_std = pred_score_arr.std(ddof=1)
#
#     print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
#     print(f"Pred Score: mean = {pred_mean:.4f}, std = {pred_std:.4f}")
#
#
# def analyze_unconditional_results():
#     disc_score_list = []
#     pred_score_list = []
#     for _ in range(20):
#         disc_score, pred_score = calculate_scores_unconditional(
#             sample_pt_path="/playpen/haochenz/ckpts_ts_generation/DiTDH-S-BS512-LR2e-4-unconditional/uncond_samples.pt",
#             real_pt_path="text2ts_results.pt"
#         )
#
#         disc_score_list.append(disc_score)
#         pred_score_list.append(pred_score)
#
#     disc_score_arr = np.array(disc_score_list)
#     pred_score_arr = np.array(pred_score_list)
#
#     disc_mean = disc_score_arr.mean()
#     disc_std = disc_score_arr.std(ddof=1)  # sample std
#
#     pred_mean = pred_score_arr.mean()
#     pred_std = pred_score_arr.std(ddof=1)
#
#     print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
#     print(f"Pred Score: mean = {pred_mean:.4f}, std = {pred_std:.4f}")



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

            emb_list.append(out.cpu().numpy())

    return np.concatenate(emb_list, axis=0)


def _calculate_fid_from_embeddings(real_emb, fake_emb, eps=1e-6):
    """
    real_emb: np.ndarray, shape (N, D)
    fake_emb: np.ndarray, shape (M, D)
    """
    mu_r = np.mean(real_emb, axis=0)
    mu_f = np.mean(fake_emb, axis=0)

    sigma_r = np.cov(real_emb, rowvar=False)
    sigma_f = np.cov(fake_emb, rowvar=False)

    if sigma_r.ndim == 0:
        sigma_r = np.array([[sigma_r]])
    if sigma_f.ndim == 0:
        sigma_f = np.array([[sigma_f]])

    sigma_r = sigma_r + eps * np.eye(sigma_r.shape[0])
    sigma_f = sigma_f + eps * np.eye(sigma_f.shape[0])

    diff = mu_r - mu_f
    covmean = sqrtm(sigma_r @ sigma_f)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean)
    return float(fid)


def calculate_all_scores(results_path, block_id):
    if block_id is not None:
        pred_start = block_id * 32
        pred_end = block_id * 32 + 32
    else:
        pred_start = 0
        pred_end = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dict = torch.load(results_path, map_location="cpu", weights_only=False)
    real = results_dict["real_ts"][:, :, pred_start:pred_end]
    print(f"real shape: {real.shape}")
    # ----------------------------
    # load MOMENT once
    # ----------------------------
    # moment_model = MOMENTPipeline.from_pretrained(
    #     "AutonLab/MOMENT-1-large",
    #     model_kwargs={"task_name": "embedding"},
    # )
    # moment_model.init()
    # moment_model = moment_model.to(device)
    # moment_model.eval()

    disc_score_list = []
    fid_score_list = []

    # real embedding can be computed once
    # real_emb = _moment_embed(moment_model, real, device)

    for i in range(10):
        fake = results_dict["sampled_ts"][i][:, :, pred_start:pred_end]

        discriminative_score = discriminative_score_metrics(
            real, fake,
            real.shape[-1],
            device,
        )
        disc_score_list.append(discriminative_score)

        # fake_emb = _moment_embed(moment_model, fake, device)
        # fid_score = _calculate_fid_from_embeddings(real_emb, fake_emb)
        # fid_score_list.append(fid_score)

    disc_score_arr = np.array(disc_score_list)
    # fid_score_arr = np.array(fid_score_list)

    disc_mean = disc_score_arr.mean()
    disc_std = disc_score_arr.std(ddof=1)

    # fid_mean = fid_score_arr.mean()
    # fid_std = fid_score_arr.std(ddof=1)

    print(results_path)
    print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
    # print(f"MOMENT-FID: mean = {fid_mean:.4f}, std = {fid_std:.4f}")
    print("---" * 50)

    return real


# def calculate_all_scores(results_path, block_id):
#     if block_id is not None:
#         pred_start = block_id * 32
#         pred_end = block_id * 32 + 32
#     else:
#         pred_start = 0
#         pred_end = 128
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     results_dict = torch.load(results_path, map_location="cpu", weights_only=False)
#     real = results_dict["real_ts"][:,:,pred_start:pred_end]
#     disc_score_list = []
#     pred_score_list = []
#     for i in range(5):
#         # print(i)
#         fake = results_dict["sampled_ts"][i][:,:,pred_start:pred_end]
#         # print(f"real: {real.shape}, fake: {fake.shape}")
#         # breakpoint()
#         discriminative_score = moment_discriminative_score_metrics(
#             real, fake,
#             real.shape[-1],
#             device,
#         )
#
#         # discriminative_score = discriminative_score_metrics(
#         #     real, fake,
#         #     real.shape[-1],
#         #     device,
#         # )
#
#         # print(f"Discriminative Score Metrics: {discriminative_score}")
#
#         # predictive_score = predictive_score_metrics(real, fake, device)
#         # # print(f"Predictive Score Metrics: {predictive_score}")
#         disc_score_list.append(discriminative_score)
#         # pred_score_list.append(predictive_score)
#
#     disc_score_arr = np.array(disc_score_list)
#     # pred_score_arr = np.array(pred_score_list)
#
#     disc_mean = disc_score_arr.mean()
#     disc_std = disc_score_arr.std(ddof=1)  # sample std
#
#     # pred_mean = pred_score_arr.mean()
#     # pred_std = pred_score_arr.std(ddof=1)
#     print(results_path)
#     print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
#     # print(f"Pred Score: mean = {pred_mean:.4f}, std = {pred_std:.4f}")
#     print("---"*50)
#     return real


# def calculate_all_scores_two_paths(real_path, fake_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # real = np.load(real_path, allow_pickle=True)
#     real_dict = torch.load(real_path, map_location="cpu", weights_only=False)
#     real = real_dict["real_ts"]
#     # real = torch.from_numpy(real).to(device)
#     print(f"real shape = {real.shape}")
#     results_dict = torch.load(fake_path, map_location="cpu", weights_only=False)
#     print(f"fake shape = {results_dict['sampled_ts'].shape}")
#     num_sampels = min(len(real), len(results_dict["sampled_ts"][0]))
#     real = real[:num_sampels]
#
#     disc_score_list = []
#     pred_score_list = []
#     for i in range(10):
#         # print(i)
#         fake = results_dict["sampled_ts"][i][:num_sampels]
#         discriminative_score = discriminative_score_metrics(
#             real, fake,
#             real.shape[-1],
#             device,
#         )
#         # print(f"Discriminative Score Metrics: {discriminative_score}")
#
#         # predictive_score = predictive_score_metrics(real, fake, device)
#         # print(f"Predictive Score Metrics: {predictive_score}")
#         disc_score_list.append(discriminative_score)
#         # pred_score_list.append(predictive_score)
#
#     disc_score_arr = np.array(disc_score_list)
#     # pred_score_arr = np.array(pred_score_list)
#
#     disc_mean = disc_score_arr.mean()
#     disc_std = disc_score_arr.std(ddof=1)  # sample std
#
#     # pred_mean = pred_score_arr.mean()
#     # pred_std = pred_score_arr.std(ddof=1)
#     print(fake_path)
#     print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
#     # print(f"Pred Score: mean = {pred_mean:.4f}, std = {pred_std:.4f}")
#     print("---"*50)
#     return real



def calculate_all_scores_two_paths(real_path, fake_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_dict = torch.load(real_path, map_location="cpu", weights_only=False)
    real = real_dict["real_ts"]
    print(f"real shape = {real.shape}")
    results_dict = torch.load(fake_path, map_location="cpu", weights_only=False)
    print(f"fake shape = {results_dict['sampled_ts'].shape}")
    num_samples = min(len(real), len(results_dict["sampled_ts"][0]))
    real = real[:num_samples]

    print(f"real shape: {real.shape}")
    # ----------------------------
    # load MOMENT once
    # ----------------------------
    moment_model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"},
    )
    moment_model.init()
    moment_model = moment_model.to(device)
    moment_model.eval()

    disc_score_list = []
    fid_score_list = []

    # real embedding can be computed once
    real_emb = _moment_embed(moment_model, real, device)

    for i in range(10):
        fake = results_dict["sampled_ts"][i, :num_samples]

        discriminative_score = discriminative_score_metrics(
            real, fake,
            real.shape[-1],
            device,
        )
        disc_score_list.append(discriminative_score)

        fake_emb = _moment_embed(moment_model, fake, device)
        fid_score = _calculate_fid_from_embeddings(real_emb, fake_emb)
        fid_score_list.append(fid_score)

    disc_score_arr = np.array(disc_score_list)
    fid_score_arr = np.array(fid_score_list)

    disc_mean = disc_score_arr.mean()
    disc_std = disc_score_arr.std(ddof=1)

    fid_mean = fid_score_arr.mean()
    fid_std = fid_score_arr.std(ddof=1)

    print(fake_path)
    print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
    print(f"MOMENT-FID: mean = {fid_mean:.4f}, std = {fid_std:.4f}")
    print("---" * 50)

    return real


def local_visualize():
    samples = torch.load("./results/synth_u_real_text_samples.pt")
    for i in range(len(samples)):
        real = samples['real_ts'][i]
        num_channels = real.shape[0]
        num_segments = 4
        for i_channel in range(num_channels):
            for i_seg in range(num_segments):
                plt.plot(real[i_channel, 32*i_seg:32*i_seg+32], label="real", color="orange")
                for j in range(10):
                    fake = samples['sampled_ts'][j, i, i_channel, 32*i_seg:32*i_seg+32]
                    plt.plot(fake, label="fake", color="blue", alpha=0.3)
                key = f"seg{i_seg + 1}_channel{i_channel}"
                cap = next(d[key] for d in samples['caption'][i] if key in d)
                plt.title(cap)
                plt.savefig(f"./results/figures/{cap}.png")
                plt.close()
        # plt.legend()
        plt.show()
        if i > 3:
            break

if __name__ == "__main__":

    # calculate_all_scores_two_paths(
    #     "/playpen/haochenz/save/causal_moment_embeds/synth_u/0/real_text_samples.pt",
    #     "/playpen/haochenz/save/causal_moment_embeds/synth_u/0/real_text_samples.pt",
    # )

    calculate_all_scores_two_paths(
        "/playpen/haochenz/save/causal_correct/synth_u/0/samples.pt",
        "/playpen/haochenz/save/causal_correct/synth_u/0/samples.pt",
    )

    calculate_all_scores_two_paths(
        "/playpen/haochenz/save/causal_correct/synth_u/0/samples.pt",
        "/playpen/haochenz/save/causal_correct/synth_u/0/fake_text_samples.pt",
    )

    calculate_all_scores_two_paths(
        "/playpen/haochenz/save/causal_correct/synth_u/0/samples.pt",
        "/playpen/haochenz/save/causal_no_text/synth_u/0/samples.pt",
    )

    # calculate_all_scores_two_paths(
    #     "/playpen/haochenz/save/causal_moment_embeds/synth_m/0/real_text_samples.pt",
    #     "/playpen/haochenz/save/causal_moment_embeds/synth_m/0/real_text_samples.pt",
    # )


    # calculate_all_scores_two_paths(
    #     "/playpen/haochenz/save/non_causal_correct/synth_u/0/samples.pt",
    #     "/playpen/haochenz/save/non_causal_correct/synth_u/0/samples.pt",
    # )
    #
    # calculate_all_scores_two_paths(
    #     "/playpen/haochenz/save/non_causal_correct/synth_m/0/samples.pt",
    #     "/playpen/haochenz/save/non_causal_correct/synth_m/0/samples.pt",
    # )
    #
    #
    #
    # calculate_all_scores_two_paths(
    #     "/playpen/haochenz/save/causal_correct/synth_u/0/samples.pt",
    #     fake_path="/playpen/haochenz/save/causal_correct/synth_u/0/fake_text_samples.pt"
    # )
    #
    # calculate_all_scores_two_paths(
    #     "/playpen/haochenz/save/causal_correct/synth_m/0/samples.pt",
    #     fake_path="/playpen/haochenz/save/causal_correct/synth_m/0/fake_text_samples.pt"
    # )
    #
    #
    # calculate_all_scores_two_paths(
    #     "/playpen/haochenz/save/causal_correct/synth_u/0/samples.pt",
    #     fake_path="/playpen/haochenz/save/non_causal_correct/synth_u/0/fake_text_samples.pt"
    # )
    #
    # calculate_all_scores_two_paths(
    #     "/playpen/haochenz/save/causal_correct/synth_m/0/samples.pt",
    #     fake_path="/playpen/haochenz/save/non_causal_correct/synth_m/0/fake_text_samples.pt"
    # )

    # all = torch.load('samples.pth', map_location="cpu")
    #
    # plt.plot(all['real_ts'][0,0,:32], label='real')
    # # plt.plot(all['sampled_ts'][0,0,:32,0], label='fake')
    # plt.show()

    # local_visualize()





