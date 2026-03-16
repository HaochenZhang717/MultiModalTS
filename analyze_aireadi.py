import torch
import numpy as np
import matplotlib.pyplot as plt



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