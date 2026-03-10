import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import copy  # 用于深度拷贝模型权重

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# Dataset
# ============================
class GlucoseDataset(Dataset):
    def __init__(self, data_path, prefix_len=64, target_len=64):
        data = np.load(data_path)
        self.data = data
        self.prefix = self.data[:, :prefix_len, :]
        self.target = self.data[:, prefix_len:prefix_len + target_len, :]

    def __len__(self):
        return len(self.prefix)

    def __getitem__(self, idx):
        return {"prefix": self.prefix[idx], "target": self.target[idx]}


# ============================
# Model
# ============================
class GRUPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, output_len=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_len)
        )

    def forward(self, x):
        _, h = self.gru(x)
        last_hidden = h[-1]
        pred = self.head(last_hidden)
        return pred.unsqueeze(-1)


# ============================
# Training Logic (带最佳权重保存)
# ============================
def train(model, train_loader, val_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())  # 初始化最佳权重

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            x, y = batch["prefix"].to(device), batch["target"].to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["prefix"].to(device), batch["target"].to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 检查是否为最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if epoch > 0:  # 避免初始打印
                print(f"--> Epoch {epoch:02d}: New best Val Loss: {best_val_loss:.6f}")

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:02d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

    # 训练结束后，加载表现最好的权重
    print(f"\nTraining complete. Loading best weights (Val Loss: {best_val_loss:.6f})")
    model.load_state_dict(best_model_wts)
    return model


# ============================
# Visualization
# ============================
def visualize(model, dataset, num_examples=3):
    model.eval()
    with torch.no_grad():
        indices = np.random.choice(len(dataset), num_examples, replace=False)
        for i in indices:
            sample = dataset[i]
            prefix = torch.from_numpy(sample["prefix"]).unsqueeze(0).to(device)
            # 反缩放回真实血糖值 (400.0 是 Dataset 里的缩放系数)
            target = sample["target"].cpu().numpy()
            pred = model(prefix).squeeze(0).cpu().numpy()
            prefix_np = prefix.cpu().numpy()

            plt.figure(figsize=(10, 4))
            plt.plot(range(64), prefix_np, label="History", color='blue')
            plt.plot(range(64, 128), target, label="True Future", color='green', linestyle='--')
            plt.plot(range(64, 128), pred, label="Best Model Prediction", color='red')
            plt.axvline(x=63, color='gray', alpha=0.5)
            plt.legend()
            plt.title(f"Best Weights Visualization - Sample {i}")
            plt.ylabel("Glucose (mg/dL)")
            plt.savefig(f"best_model_example_{i}.png")
            plt.close()


# ============================
# Main
# ============================
def main():
    data_path = "glucose_single_patient.npy"

    full_dataset = GlucoseDataset(data_path, prefix_len=192, target_len=64)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = GRUPredictor().to(device)

    # 这里的 model 会在训练结束后自动变为最佳权重状态
    model = train(model, train_loader, val_loader, epochs=1)

    visualize(model, val_ds)


if __name__ == "__main__":
    main()