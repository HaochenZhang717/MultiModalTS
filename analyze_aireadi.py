import torch
import numpy as np
import matplotlib.pyplot as plt


import torch
import numpy as np


def calculate_forecast_scores(results_path, pred_start, pred_end):

    results_dict = torch.load(results_path, map_location="cpu", weights_only=False)

    real = results_dict["real_ts"][:, :, pred_start:pred_end]
    print(f"real shape: {real.shape}")

    preds = results_dict["sampled_ts"][:, :, :, pred_start:pred_end]
    print(f"pred shape: {preds.shape}")

    # preds shape
    # [num_samples, B, C, T]
    preds = preds.numpy()
    real = real.numpy()

    # --------------------------------------------------
    # point forecast (median)
    # --------------------------------------------------

    median_pred = np.median(preds, axis=0)

    mse = np.mean((median_pred - real) ** 2)
    mae = np.mean(np.abs(median_pred - real))
    rmse = np.sqrt(mse)

    # --------------------------------------------------
    # probabilistic metrics
    # --------------------------------------------------

    # CRPS approximation
    crps = np.mean(np.abs(preds - real[None, ...])) \
           - 0.5 * np.mean(np.abs(preds[:, None, ...] - preds[None, :, ...]))

    # prediction interval (90%)
    lower = np.percentile(preds, 5, axis=0)
    upper = np.percentile(preds, 95, axis=0)

    # coverage
    inside = (real >= lower) & (real <= upper)
    picp = inside.mean()

    # interval width
    mpiw = np.mean(upper - lower)

    print(results_path)

    print(f"MSE  : {mse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")

    print(f"CRPS : {crps:.6f}")
    print(f"PICP : {picp:.4f}")
    print(f"MPIW : {mpiw:.6f}")

    print("---" * 50)

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "CRPS": crps,
        "PICP": picp,
        "MPIW": mpiw,
    }



if __name__ == "__main__":
    samples = torch.load("/Users/zhc/Documents/LitsDatasets/samples.pt")

    for i in range(len(samples)):
        real = samples['real_ts'][i].flatten()
        plt.plot(real, label="real", color="orange")
        for j in range(10):
            fake = samples['sampled_ts'][j,i].flatten()
            plt.plot(fake, label="fake", color="blue")
        # plt.legend()
        plt.show()
        if i > 3:
            break
    # calculate_forecast_scores(
    #     results_path="/playpen/haochenz/save/aireadi/retinal_text_history_0315/0/samples.pt",
    #     pred_start = 768,
    #     pred_end = 1024,
    # )
    #
    # calculate_forecast_scores(
    #     results_path="/playpen/haochenz/save/aireadi/history_0315/0/samples.pt",
    #     pred_start=768,
    #     pred_end=1024,
    # )