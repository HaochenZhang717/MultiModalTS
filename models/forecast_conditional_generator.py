import torch
import torch.nn as nn

from models.encoders.attr_encoder import AttributeEncoder
from models.encoders.text_encoder import TextEncoder, CLIPTextEncoder, MultiModalEncoder
from models.encoders.cond_projector import TextProjectorMVarMScaleMStep, AttrProjectorAvg
from models.forecast_unconditional_generator import UnConditionalPredictor
from models.cttp.cttp_model import CTTP
import time
import random
import yaml





class ConditionalPredictor(nn.Module):
    def __init__(self, diff_configs, cond_configs):
        super().__init__()
        self.device = diff_configs["device"] if torch.cuda.is_available() else "cpu"
        self.diff_configs = diff_configs
        self.cond_configs = cond_configs
        self._init_condition_encoders(diff_configs, cond_configs)
        self._init_diff(diff_configs)

    def _init_condition_encoders(self, diff_configs, cond_configs):
        if cond_configs["cond_modal"] == "aireadi":
            cond_configs["aireadi"]["device"] = self.device
            self.cond_projector = nn.Sequential(
                nn.Linear(cond_configs["aireadi"]["pretrain_model_dim"], cond_configs["aireadi"]["vl_emb_hidden_dim"]),
                nn.LayerNorm(cond_configs["aireadi"]["vl_emb_hidden_dim"]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(cond_configs["aireadi"]["vl_emb_hidden_dim"], cond_configs["aireadi"]["vl_emb"])
            )
            self.cond_projector = self.cond_projector.to(self.device)
        else:
            raise NotImplementedError

    def _init_diff(self, configs):
        configs["device"] = self.device

        self.generator = UnConditionalPredictor(configs=configs)
        if configs["generator_pretrain_path"] != "":
            self.generator.load_state_dict(torch.load(configs["generator_pretrain_path"]))
            print("Load the pretrain unconditional generator")
        else:
            print("Learn from scratch")

    def forward(self, batch, is_train):
        x, tp, attr_embed = self._unpack_data_cond_gen(batch)
        B, _, T = x.shape

        PREDICT_START = 768
        PREDICT_END = 1024


        # here we test only predict the first block
        attn_mask = torch.zeros((B, T)).to(x.device)  # (B ,T)
        attn_mask[:, :PREDICT_END] = 1

        loss_mask = torch.zeros((B, T)).to(x.device)  # (B ,T)
        loss_mask[:, PREDICT_START:PREDICT_END] = 1

        B = x.shape[0]
        if is_train:
            t = torch.randint(0, self.generator.num_steps, [B], device=self.device)

            attr_embed = self.cond_projector(attr_embed)  # for now we are not using projector.

            loss = self.generator._noise_estimation_loss(x, tp, attr_embed, t, prefix_length=PREDICT_START)
            return loss
        
        loss_dict = {}
        attr_embed = self.cond_projector(attr_embed)
        for t in range(self.generator.num_steps):
            t = (torch.ones(B, device=self.device) * t).long()

            tmp_loss_dict = self.generator._noise_estimation_loss(x, tp, attr_embed, t, prefix_length=PREDICT_START)
            for k in tmp_loss_dict:
                if k in loss_dict.keys():
                    loss_dict[k] += tmp_loss_dict[k]
                else:
                    loss_dict[k] = tmp_loss_dict[k]
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / self.generator.num_steps
        return loss_dict

    def _unpack_data_cond_gen(self, batch):
        ts = batch["glucose_window"].to(self.device).float()  # batch_size, num_channels, seq_len
        B, _, T = ts.shape
        tp = torch.arange(T).repeat(B, 1).to(self.device).float()
        text_embed = batch["text_embedding"].to(self.device).float()
        retinal_embed = batch["retinal_embedding"].to(self.device).float()
        attr_embed = torch.cat((retinal_embed, text_embed), dim=1).mean(1, keepdim=True)
        return ts, tp, attr_embed

    @torch.no_grad()
    def generate(self, batch, n_samples, sampler="ddim"):

        PREDICT_START = 768
        PREDICT_END = 1024

        ts, tp, attr_embed = self._unpack_data_cond_gen(batch)

        samples = []
        B, _, T = ts.shape

        for i in range(n_samples):
            x = torch.randn_like(ts)
            x[:,:,:PREDICT_START] = ts[:,:,:PREDICT_START]
            attr_emb = self.cond_projector(attr_embed)

            for t in range(self.generator.num_steps-1, -1, -1):
                noise = torch.randn_like(x)
                t = (torch.ones(B, device=self.device) * t).long()
                pred_noise, _ = self.generator.predict_noise(x, tp, attr_emb, t)
                if sampler == "ddpm":
                    x = self.generator.ddpm.reverse(x, pred_noise, t, noise)
                else:
                    x = self.generator.ddim.reverse(x, pred_noise, t, noise, is_determin=True)

                x[:, :, :PREDICT_START] = ts[:, :, :PREDICT_START]

            samples.append(x)
        return torch.stack(samples)
