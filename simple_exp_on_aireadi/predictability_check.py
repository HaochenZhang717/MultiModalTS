import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# Dataset with Scaling
# ============================

class GlucoseDataset(Dataset):
    def __init__(self, data_path, prefix_len=64, target_len=64):
        # Load data: assuming shape (N, Total_Len, 1)
        data = np.load(data_path)

        # Simple Min-Max Scaling (assuming glucose range ~40-400)
        # It's better to scale based on the training set, but this is a quick fix
        self.data = torch.tensor(data).float()
        # print(self.data.max())
        # print(self.data.min())
        self.prefix = self.data[:, :prefix_len, :]
        self.target = self.data[:, prefix_len:prefix_len + target_len, :]

    def __len__(self):
        return len(self.prefix)

    def __getitem__(self, idx):
        return {
            "prefix": self.prefix[idx],
            "target": self.target[idx]
        }


# ============================
# Improved Model Architecture
# ============================

class GRUPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, output_len=64):
        super().__init__()
        self.output_len = output_len

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_len)  # Map hidden state to the whole future window
        )

    def forward(self, x):
        # x shape: [batch, seq_len, 1]
        _, h = self.gru(x)

        # Take the hidden state of the last layer: [batch, hidden_dim]
        last_hidden = h[-1]

        # Predict the full sequence: [batch, output_len]
        pred = self.head(last_hidden)

        # Reshape to [batch, output_len, 1] to match dataset targets
        return pred.unsqueeze(-1)


# ============================
# Training Logic
# ============================

def train(model, train_loader, val_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch["prefix"].to(device)
            y = batch["target"].to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["prefix"].to(device), batch["target"].to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:02d} | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f}")


# ============================
# Visualization
# ============================

def visualize(model, dataset, num_examples=3):
    model.eval()
    with torch.no_grad():
        indices = np.random.choice(len(dataset), num_examples, replace=False)
        for i in indices:
            sample = dataset[i]
            prefix = sample["prefix"].unsqueeze(0).to(device)
            target = sample["target"].cpu().numpy() * 400.0  # Unscale for plotting

            pred = model(prefix).squeeze(0).cpu().numpy() * 400.0  # Unscale
            prefix_np = sample["prefix"].cpu().numpy() * 400.0  # Unscale

            plt.figure(figsize=(10, 4))
            plt.plot(range(len(prefix_np)), prefix_np, label="Input (History)", color='blue')

            # Target and Prediction start where history ends
            time_axis = range(len(prefix_np), len(prefix_np) + len(target))
            plt.plot(time_axis, target, label="True Future", color='green', linestyle='--')
            plt.plot(time_axis, pred, label="Model Prediction", color='red')

            plt.axvline(x=len(prefix_np) - 1, color='gray', linestyle=':')
            plt.legend()
            plt.title(f"Glucose Prediction Example {i}")
            plt.ylabel("Glucose (mg/dL)")
            # plt.show()
            plt.savefig(f"glucose_prediction_example_{i}.png")


# ============================
# Main
# ============================

def main():
    # 1. Load Data
    data_path = "glucose_single_patient.npy"
    # Create dummy data if file doesn't exist for testing purposes
    try:
        full_dataset = GlucoseDataset(data_path)
    except FileNotFoundError:
        print("Data file not found. Creating dummy data...")
        dummy_data = np.random.randn(100, 128, 1) * 20 + 120
        np.save(data_path, dummy_data)
        full_dataset = GlucoseDataset(data_path)

    # 2. Split into Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # 3. Init Model
    model = GRUPredictor(output_len=64).to(device)

    # 4. Run Loop
    train(model, train_loader, val_loader, epochs=100)
    visualize(model, val_ds)


if __name__ == "__main__":
    main()