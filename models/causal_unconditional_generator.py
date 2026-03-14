import torch
import torch.nn as nn

from models.diffusion.causal_verbalts import CausalVerbalTS
from samplers import CausalDDPMSampler, DDIMSampler
import numpy as np
import time
import random

class CausalUnConditionalGenerator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.device = configs["device"]
        self.configs = configs
        self._init_diff(configs["diffusion"])

    def _init_diff(self, configs):
        configs["device"] = self.device
        self.diff_model = CausalVerbalTS(configs, inputdim=1).to(self.device)

        # self.diff_model = DiTModel(configs).to(self.device)
        self.num_steps = configs["num_steps"]
        self.ddpm = CausalDDPMSampler(self.num_steps, configs["beta_start"], configs["beta_end"], configs["schedule"], self.device)
        self.ddim = DDIMSampler(self.num_steps, configs["beta_start"], configs["beta_end"], configs["schedule"], self.device)
    
    def _noise_estimation_loss(self, x, tp, text_embed, t, loss_mask, attn_mask):
        noise = torch.randn_like(x)

        noise_mask = attn_mask - loss_mask.unsqueeze(1)
        noisy_x = self.ddpm.forward(x, t, noise, noise_mask)


        pred_noise, loss_dict = self.predict_noise(noisy_x, tp, text_embed, t, attn_mask)
        residual = noise - pred_noise

        mask = loss_mask.unsqueeze(1)  # (B,1,T)
        loss_dict["noise_loss"] = ((residual ** 2) * mask).sum() / mask.sum()

        all_loss = torch.zeros_like(loss_dict["noise_loss"])
        for k in loss_dict.keys():
            all_loss += loss_dict[k]
        loss_dict["all"] = all_loss
        return loss_dict
    
    """
    Pretrain.
    """
    def forward(self, batch, is_train):
        raise NotImplementedError
        # x, tp = self._unpack_data_uncond_gen(batch)
        # B = x.shape[0]
        #
        # if is_train:
        #     t = torch.randint(0, self.num_steps, [B], device=self.device)
        #     loss = self._noise_estimation_loss(x, tp, None, t)
        #     return loss
        #
        # loss_dict = {}
        # for t in range(self.num_steps):
        #     t = (torch.ones(B, device=self.device) * t).long()
        #     tmp_loss_dict = self._noise_estimation_loss(x, tp, None, t)
        #     for k in tmp_loss_dict:
        #         if k in loss_dict.keys():
        #             loss_dict[k] += tmp_loss_dict[k]
        #         else:
        #             loss_dict[k] = tmp_loss_dict[k]
        # for k in loss_dict:
        #     loss_dict[k] = loss_dict[k] / self.num_steps
        # return loss_dict

    def _unpack_data_uncond_gen(self, batch):
        raise NotImplementedError
        # ts = batch["ts"].to(self.device).float()
        # tp = batch["tp"].to(self.device).float()
        # ts = ts.permute(0, 2, 1)
        #
        # return ts, tp

    """
    Generation.
    """
    @torch.no_grad()
    def generate(self, batch, n_samples, sampler="ddim"):
        raise NotImplementedError
        # ts, tp = self._unpack_data_uncond_gen(batch)
        # samples = []
        # B = ts.shape[0]
        # for i in range(n_samples):
        #     x = torch.randn_like(ts)
        #     for t in range(self.num_steps-1, -1, -1):
        #         noise = torch.randn_like(x)
        #         t = (torch.ones(B, device=self.device) * t).long()
        #         pred_noise, _ = self.predict_noise(x, tp, None, t)
        #         if sampler == "ddpm":
        #             x = self.ddpm.reverse(x, pred_noise, t, noise)
        #         else:
        #             x = self.ddim.reverse(x, pred_noise, t, noise, is_determin=True)
        #     samples.append(x)
        # return torch.stack(samples)

    def predict_noise(self, xt, tp, text_embed, t, attn_mask):
        noisy_x = torch.unsqueeze(xt, 1)
        pred_noise, loss_dict = self.diff_model(noisy_x, tp, text_embed, t, attn_mask)
        return pred_noise, loss_dict