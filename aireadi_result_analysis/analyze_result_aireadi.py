import torch
import numpy as np
import matplotlib.pyplot as plt
import random


def evaluate_prediction(results, save_dir=None, num_visualize=5):
    real_ts = results["real_ts"]  # (N,1,T)
    sampled_ts = results["sampled_ts"]  # (S,N,T,1)

    # reshape
    real_ts = real_ts.squeeze(1)  # (N,T)
    sampled_ts = sampled_ts.squeeze(-1)  # (S,N,T)

    S, N, T = sampled_ts.shape

    assert T >= 1024 or T >= 512, "sequence length seems wrong"

    # ==========================
    # split observed / future
    # ==========================
    obs_len = 512

    real_future = real_ts[:, obs_len:]  # (N, future)
    pred_future = sampled_ts[:, :, obs_len:]  # (S,N,future)

    real_future_exp = real_future.unsqueeze(0)

    # ==========================
    # metrics
    # ==========================
    mse = ((pred_future - real_future_exp) ** 2).mean()

    mae = (pred_future - real_future_exp).abs().mean()

    mse_per_sample = ((pred_future - real_future_exp) ** 2).mean(dim=(1, 2))

    # best-of-K
    mse_each = ((pred_future - real_future_exp) ** 2).mean(dim=2)  # (S,N)
    best_mse = mse_each.min(dim=0)[0].mean()

    print("========== Metrics ==========")
    print("MSE:", mse.item())
    print("MAE:", mae.item())
    print("Best-of-K MSE:", best_mse.item())
    print("=============================")

    # ==========================
    # visualization
    # ==========================

    indices = random.sample(range(N), num_visualize)

    for idx in indices:

        real = real_ts[idx].cpu().numpy()
        preds = sampled_ts[:, idx].cpu().numpy()

        plt.figure(figsize=(10, 4))

        # observed
        plt.plot(range(obs_len), real[:obs_len], label="observed", linewidth=3)

        # ground truth
        plt.plot(range(obs_len, T), real[obs_len:], label="ground truth", linewidth=3)

        # predictions
        for s in range(S):
            plt.plot(range(obs_len, T), preds[s, obs_len:], alpha=0.3, color="red")

        plt.axvline(obs_len, linestyle="--", color="black")

        plt.legend()
        plt.title(f"Forecast example #{idx}")

        if save_dir:
            plt.savefig(f"{save_dir}/forecast_{idx}.png", bbox_inches="tight")

        plt.show()

    # ==========================
    # fan chart example
    # ==========================

    idx = indices[0]

    real = real_ts[idx].cpu().numpy()
    preds = sampled_ts[:, idx].cpu().numpy()

    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    plt.figure(figsize=(10, 4))

    plt.plot(real[:obs_len], label="observed", linewidth=3)
    plt.plot(range(obs_len, T), real[obs_len:], label="ground truth", linewidth=3)

    plt.plot(range(obs_len, T), mean_pred[obs_len:], label="mean prediction")

    plt.fill_between(
        range(obs_len, T),
        mean_pred[obs_len:] - std_pred[obs_len:],
        mean_pred[obs_len:] + std_pred[obs_len:],
        alpha=0.3,
        label="uncertainty"
    )

    plt.axvline(obs_len, linestyle="--", color="black")

    plt.legend()
    plt.title("Forecast fan chart")

    if save_dir:
        plt.savefig(f"{save_dir}/fan_chart.png", bbox_inches="tight")

    plt.show()


# ==========================
# run
# ==========================
if __name__ == "__main__":
    results = torch.load("/Users/zhc/Documents/samples.pth")
    evaluate_prediction(results)

