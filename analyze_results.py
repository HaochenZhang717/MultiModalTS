import torch
import matplotlib.pyplot as plt
import numpy as np
from metrics.discriminative_torch import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics


def visualize_sample(pt_path, idx=0):
    data = torch.load(pt_path, map_location="cpu")

    sample_dict = data[idx]
    caption = sample_dict["caption"]
    ts = sample_dict["sample"]  # Tensor
    real = sample_dict["orig"]

    if isinstance(ts, torch.Tensor):
        ts = ts.numpy()

    print("Caption:")
    print(caption)
    print("Shape:", ts.shape)
    print("========"*30)

    plt.figure(figsize=(10, 4))

    # 如果是单通道
    if ts.ndim == 1:
        plt.plot(ts)
    elif ts.ndim == 2:
        # 多通道，每个通道画一条
        for c in range(ts.shape[0]):
            plt.plot(ts[c], label=f"Sample Channel {c}")

    else:
        raise ValueError("Unsupported time series shape")


    if real.ndim == 1:
        plt.plot(real)
    elif real.ndim == 2:
        # 多通道，每个通道画一条
        for c in range(real.shape[0]):
            plt.plot(real[c], label=f"real Channel {c}")

    else:
        raise ValueError("Unsupported time series shape")

    plt.legend()
    plt.title("Generated Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def calculate_scores_from_real_language(pt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(pt_path, map_location=device)
    real = []
    fake = []
    for datum in data:
        real.append(datum["orig"])
        fake.append(datum["sample"])
    real = torch.stack(real)
    fake = torch.stack(fake)

    real = real.permute(0, 2, 1)
    fake = fake.permute(0, 2, 1)
    print("Real shape:", real.shape)
    print("Fake shape:", fake.shape)
    discriminative_score = discriminative_score_metrics(
        real, fake,
        real.shape[-1],
        device,
    )
    print(f"Discriminative Score Metrics: {discriminative_score}")

    predictive_score = predictive_score_metrics(
        real, fake, device
    )
    print(f"Predictive Score Metrics: {predictive_score}")

    return (
        discriminative_score,
        predictive_score,
    )


def calculate_scores_from_prior_language(sample_pt_path, real_pt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_data = torch.load(sample_pt_path, map_location=device)
    fake = []
    for sample_datum in sample_data:
        fake.append(sample_datum["sample"])
    fake = torch.stack(fake)

    real_data = torch.load(real_pt_path, map_location=device)
    real = []
    for real_datum in real_data:
        real.append(real_datum["orig"])
    real = torch.stack(real)


    real = real.permute(0, 2, 1)
    fake = fake.permute(0, 2, 1)
    print("Real shape:", real.shape)
    print("Fake shape:", fake.shape)
    discriminative_score = discriminative_score_metrics(
        real, fake,
        real.shape[-1],
        device,
    )
    print(f"Discriminative Score Metrics: {discriminative_score}")

    predictive_score = predictive_score_metrics(
        real, fake, device
    )
    print(f"Predictive Score Metrics: {predictive_score}")

    return (
        discriminative_score,
        predictive_score,
    )


def calculate_scores_unconditional(sample_pt_path, real_pt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fake = torch.load(sample_pt_path, map_location=device)

    real_data = torch.load(real_pt_path, map_location=device)
    real = []
    for real_datum in real_data:
        real.append(real_datum["orig"])
    real = torch.stack(real)


    real = real.permute(0, 2, 1)
    fake = fake.permute(0, 2, 1)
    print("Real shape:", real.shape)
    print("Fake shape:", fake.shape)
    discriminative_score = discriminative_score_metrics(
        real, fake,
        real.shape[-1],
        device,
    )
    print(f"Discriminative Score Metrics: {discriminative_score}")

    predictive_score = predictive_score_metrics(
        real, fake, device
    )
    print(f"Predictive Score Metrics: {predictive_score}")

    return (
        discriminative_score,
        predictive_score,
    )


def analyze_text2ts_results():
    disc_score_list = []
    pred_score_list = []
    for _ in range(20):
        # disc_score, pred_score = calculate_scores_from_prior_language(
        #     sample_pt_path="/playpen/haochenz/ckpts_ts_generation/DiTDH-S/text2ts_from_language_prior_results.pt",
        #     real_pt_path="/playpen/haochenz/ckpts_ts_generation/DiTDH-S/text2ts_results.pt"
        # )
        disc_score, pred_score = calculate_scores_from_real_language(
            pt_path="/playpen/haochenz/ckpts_ts_generation/DiTDH-S/text2ts_results.pt")

        disc_score_list.append(disc_score)
        pred_score_list.append(pred_score)

    disc_score_arr = np.array(disc_score_list)
    pred_score_arr = np.array(pred_score_list)

    disc_mean = disc_score_arr.mean()
    disc_std = disc_score_arr.std(ddof=1)  # sample std

    pred_mean = pred_score_arr.mean()
    pred_std = pred_score_arr.std(ddof=1)

    print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
    print(f"Pred Score: mean = {pred_mean:.4f}, std = {pred_std:.4f}")


def analyze_text2ts_results_from_prior_language():
    disc_score_list = []
    pred_score_list = []
    for _ in range(20):
        disc_score, pred_score = calculate_scores_from_prior_language(
            sample_pt_path="/playpen/haochenz/ckpts_ts_generation/DiTDH-S/text2ts_from_language_prior_results.pt",
            real_pt_path="/playpen/haochenz/ckpts_ts_generation/DiTDH-S/text2ts_results.pt"
        )

        disc_score_list.append(disc_score)
        pred_score_list.append(pred_score)

    disc_score_arr = np.array(disc_score_list)
    pred_score_arr = np.array(pred_score_list)

    disc_mean = disc_score_arr.mean()
    disc_std = disc_score_arr.std(ddof=1)  # sample std

    pred_mean = pred_score_arr.mean()
    pred_std = pred_score_arr.std(ddof=1)

    print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
    print(f"Pred Score: mean = {pred_mean:.4f}, std = {pred_std:.4f}")


def analyze_unconditional_results():
    disc_score_list = []
    pred_score_list = []
    for _ in range(20):
        disc_score, pred_score = calculate_scores_unconditional(
            sample_pt_path="/playpen/haochenz/ckpts_ts_generation/DiTDH-S-BS512-LR2e-4-unconditional/uncond_samples.pt",
            real_pt_path="text2ts_results.pt"
        )

        disc_score_list.append(disc_score)
        pred_score_list.append(pred_score)

    disc_score_arr = np.array(disc_score_list)
    pred_score_arr = np.array(pred_score_list)

    disc_mean = disc_score_arr.mean()
    disc_std = disc_score_arr.std(ddof=1)  # sample std

    pred_mean = pred_score_arr.mean()
    pred_std = pred_score_arr.std(ddof=1)

    print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
    print(f"Pred Score: mean = {pred_mean:.4f}, std = {pred_std:.4f}")


def calculate_all_scores(results_path, block_id):
    if block_id is not None:
        pred_start = block_id * 32
        pred_end = block_id * 32 + 32
    else:
        pred_start = 0
        pred_end = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dict = torch.load(results_path, map_location="cpu", weights_only=False)
    real = results_dict["real_ts"][:,:,pred_start:pred_end]
    disc_score_list = []
    pred_score_list = []
    for i in range(10):
        # print(i)
        fake = results_dict["sampled_ts"][i][:,:,pred_start:pred_end]
        # print(f"real: {real.shape}, fake: {fake.shape}")
        # breakpoint()
        discriminative_score = discriminative_score_metrics(
            real, fake,
            real.shape[-1],
            device,
        )
        # print(f"Discriminative Score Metrics: {discriminative_score}")

        # predictive_score = predictive_score_metrics(real, fake, device)
        # # print(f"Predictive Score Metrics: {predictive_score}")
        disc_score_list.append(discriminative_score)
        # pred_score_list.append(predictive_score)

    disc_score_arr = np.array(disc_score_list)
    # pred_score_arr = np.array(pred_score_list)

    disc_mean = disc_score_arr.mean()
    disc_std = disc_score_arr.std(ddof=1)  # sample std

    # pred_mean = pred_score_arr.mean()
    # pred_std = pred_score_arr.std(ddof=1)
    print(results_path)
    print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
    # print(f"Pred Score: mean = {pred_mean:.4f}, std = {pred_std:.4f}")
    print("---"*50)
    return real


def calculate_all_scores_two_paths(real_path, fake_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real = np.load(real_path, allow_pickle=True)
    real = torch.from_numpy(real).to(device)
    print(f"real shape = {real.shape}")
    results_dict = torch.load(fake_path, map_location="cpu", weights_only=False)
    print(f"fake shape = {results_dict['sampled_ts'].shape}")

    disc_score_list = []
    pred_score_list = []
    for i in range(10):
        print(i)
        fake = results_dict["sampled_ts"][i]
        discriminative_score = discriminative_score_metrics(
            real, fake,
            real.shape[-1],
            device,
        )
        print(f"Discriminative Score Metrics: {discriminative_score}")

        predictive_score = predictive_score_metrics(real, fake, device)
        print(f"Predictive Score Metrics: {predictive_score}")
        disc_score_list.append(discriminative_score)
        pred_score_list.append(predictive_score)

    disc_score_arr = np.array(disc_score_list)
    pred_score_arr = np.array(pred_score_list)

    disc_mean = disc_score_arr.mean()
    disc_std = disc_score_arr.std(ddof=1)  # sample std

    pred_mean = pred_score_arr.mean()
    pred_std = pred_score_arr.std(ddof=1)
    print(fake_path)
    print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
    print(f"Pred Score: mean = {pred_mean:.4f}, std = {pred_std:.4f}")
    print("---"*50)
    return real



if __name__ == "__main__":
    # calculate_all_scores(
    #     "/playpen/haochenz/save/synth_u_causal/0314_random_batch_block/0/samples_block0",
    #     block_id=0
    # )
    #
    # calculate_all_scores(
    #     "/playpen/haochenz/save/synth_u_causal/0314_random_batch_block/0/samples_block1",
    #     block_id=1
    # )
    #
    # calculate_all_scores(
    #     "/playpen/haochenz/save/synth_u_causal/0314_random_batch_block/0/samples_block2",
    #     block_id=2
    # )

    calculate_all_scores(
        "/playpen/haochenz/save/synth_u_causal/full_train_random_batch_block/0/samples_block3",
        block_id=3
    )

    # calculate_all_scores(
    #     "/playpen/haochenz/save/synth_u_causal/full_train_random_batch_block/0/samples_all_blocks",
    #     block_id=None
    # )

    calculate_all_scores(
        "/playpen/haochenz/save/synth_u_causal/full_train_random_batch_block/0/samples_all_blocks",
        block_id=0
    )

    calculate_all_scores(
        "/playpen/haochenz/save/synth_u_causal/full_train_random_batch_block/0/samples_block0.pt",
        block_id=0
    )


    # all = torch.load('samples.pth', map_location="cpu")
    #
    # plt.plot(all['real_ts'][0,0,:32], label='real')
    # # plt.plot(all['sampled_ts'][0,0,:32,0], label='fake')
    # plt.show()







