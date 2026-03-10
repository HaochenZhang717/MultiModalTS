import numpy as np
import pandas as pd


def build_single_patient_windows(
    parquet_path,
    patient_idx=0,
    window_size=128,
    stride=32,
):

    # 读取 parquet
    df = pd.read_parquet(parquet_path)

    # 取一个病人的 glucose 序列
    seq = df.iloc[patient_idx]["glucose"]

    seq = np.array(seq, dtype=np.float32)

    print("Original sequence length:", len(seq))

    windows = []

    for start in range(0, len(seq) - window_size + 1, stride):

        window = seq[start:start + window_size]

        windows.append(window)

    windows = np.stack(windows)

    # 转成 diffusion 常用 shape (N, T, C)
    windows = windows[..., None]

    print("Sliding windows shape:", windows.shape)

    return windows


if __name__ == "__main__":

    parquet_path = "/Users/zhc/Documents/AI-READI/glucose_train.parquet"

    data = build_single_patient_windows(
        parquet_path,
        patient_idx=0,
        window_size=256,
        stride=1,
    )
    print(data.max())
    print(data.min())
    data = (data - data.min()) / (data.max() - data.min())
    np.save("glucose_single_patient.npy", data)

    print("Saved to glucose_single_patient.npy")