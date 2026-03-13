import torch
import torch.nn as nn

from models.encoders.attr_encoder import AttributeEncoder
from models.encoders.text_encoder import TextEncoder, CLIPTextEncoder, MultiModalEncoder
from models.encoders.cond_projector import TextProjectorMVarMScaleMStep, AttrProjectorAvg
from models.causal_unconditional_generator import CausalUnConditionalGenerator
from models.cttp.cttp_model import CTTP
import time
import random
import yaml

class CausalConditionalGenerator(nn.Module):
    def __init__(self, diff_configs, cond_configs):
        super().__init__()
        self.device = diff_configs["device"] if torch.cuda.is_available() else "cpu"
        self.diff_configs = diff_configs
        self.cond_configs = cond_configs
        self._init_condition_encoders(diff_configs, cond_configs)
        self._init_diff(diff_configs)

    def _init_condition_encoders(self, diff_configs, cond_configs):
        if cond_configs["cond_modal"] == "multimodal":
            cond_configs["multimodal"]["device"] = self.device
            self.cond_projector = nn.Sequential(
                nn.Linear(cond_configs["multimodal"]["pretrain_model_dim"], cond_configs["multimodal"]["vl_emb_hidden_dim"]),
                nn.LayerNorm(cond_configs["multimodal"]["vl_emb_hidden_dim"]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(cond_configs["multimodal"]["vl_emb_hidden_dim"], cond_configs["multimodal"]["vl_emb"])
            )
            self.cond_projector = self.cond_projector.to(self.device)
        else:
            raise NotImplementedError

    def _init_diff(self, configs):
        configs["device"] = self.device

        self.generator = CausalUnConditionalGenerator(configs=configs)
        if configs["generator_pretrain_path"] != "":
            self.generator.load_state_dict(torch.load(configs["generator_pretrain_path"]))
            print("Load the pretrain unconditional generator")
        else:
            print("Learn from scratch")

    def forward(self, batch, is_train):
        # x, tp, attrs, attrs_embed_batch, loss_mask = self._unpack_data_cond_gen(batch)
        x, tp, text_embed, loss_mask, attn_mask = self._unpack_data_cond_gen(batch)

        B = x.shape[0]
        if is_train:
            t = torch.randint(0, self.generator.num_steps, [B], device=self.device)

            text_embed = self.cond_projector(text_embed)  # for now we are not using projector.

            loss = self.generator._noise_estimation_loss(x, tp, text_embed, t, loss_mask, attn_mask)
            return loss
        
        loss_dict = {}
        text_embed = self.cond_projector(text_embed)
        for t in range(self.generator.num_steps):
            t = (torch.ones(B, device=self.device) * t).long()

            tmp_loss_dict = self.generator._noise_estimation_loss(x, tp, text_embed, t, loss_mask, attn_mask)
            for k in tmp_loss_dict:
                if k in loss_dict.keys():
                    loss_dict[k] += tmp_loss_dict[k]
                else:
                    loss_dict[k] = tmp_loss_dict[k]
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / self.generator.num_steps
        return loss_dict

    def _unpack_data_cond_gen(self, batch):
        ts = batch["ts"].to(self.device).float() # batch_size, num_channels, seq_len
        B, _, T  = ts.shape
        tp = torch.arange(T).repeat(B, 1).to(self.device).float()
        loss_mask = batch["loss_mask"].to(self.device).float()
        text_embed = batch["text_embedding"].to(self.device).float()
        attn_mask = batch["attn_mask"].to(self.device).float()
        return ts, tp, text_embed, loss_mask, attn_mask

    def generate(self, batch, n_samples, sampler="ddim"):
        if self.cond_configs["cond_modal"] == "constraint":
            raise ValueError("Not Changed for precomputed attr_embed yet")
            # return self.generate_constraint(batch, n_samples, sampler)
        else:
            return self.generate_text(batch, n_samples, sampler)

    @torch.no_grad()
    def generate_text(self, batch, n_samples, sampler="ddim"):
        ts, tp, text_embed, _, _ = self._unpack_data_cond_gen(batch)

        samples = []
        B, _, T = ts.shape
        num_segments = self.diff_configs["num_segments"]
        assert T % num_segments == 0
        segment_length = T // num_segments

        text_embed = self.cond_projector(text_embed)

        for i in range(n_samples):
            for causal_step in range(num_segments):
                # for each causal step, we need to first construct loss_mask and attn_mask,
                attn_mask = torch.zeros_like(ts).sum(1)
                attn_mask[:, :(causal_step+1)*segment_length] = 1.0

                loss_mask = torch.zeros_like(ts).sum(1)
                loss_mask[:, causal_step*segment_length:(causal_step+1)*segment_length] = 1.0
                loss_mask = loss_mask.unsqueeze(1)

                x = torch.randn_like(ts)
                x = x * loss_mask + ts * (1 - loss_mask)


                for t in range(self.generator.num_steps-1, -1, -1):
                    noise = torch.randn_like(x)
                    t = (torch.ones(B, device=self.device) * t).long()

                    pred_noise, _ = self.generator.predict_noise(x, tp, text_embed, t, attn_mask)
                    if sampler == "ddpm":
                        raise NotImplementedError
                        # x_pred = self.generator.ddpm.reverse(x, pred_noise, t, noise)
                    else:
                        x_pred = self.generator.ddim.reverse(x, pred_noise, t, noise, is_determin=True)

                    x = x_pred * loss_mask + ts * (1 - loss_mask)


                samples.append(x)
        return torch.stack(samples)
    
    def generate_constraint(self, batch, n_samples, sampler="ddim"):
        raise NotImplementedError
        # ts, tp, attrs, attrs_embed_batch, loss_mask = self._unpack_data_cond_gen(batch)#todo: need to change maybe
        # samples = []
        # B = ts.shape[0]
        # for i in range(n_samples):
        #     x = torch.randn_like(ts)
        #     for t in range(self.generator.num_steps-1, -1, -1):
        #         noise = torch.randn_like(x)
        #         t = (torch.ones(B, device=self.device) * t).long()
        #         with torch.no_grad():
        #             pred_noise, _ = self.generator.predict_noise(x, tp, None, t)
        #         if sampler == "ddpm":
        #             x = self.generator.ddpm.reverse(x, pred_noise, t, noise)
        #         else:
        #             x0 = self.generator.ddim.predict_x0(x, pred_noise, t).permute(0,2,1)
        #             with torch.set_grad_enabled(True):
        #                 x0.requires_grad = True
        #                 ts_emb = self.cond_guide_model.get_ts_coemb(x0, None)
        #                 text_emb = self.cond_guide_model.get_text_coemb(attrs, None)
        #                 negative_cttp = -torch.mm(ts_emb, text_emb.permute(1,0)).trace()
        #                 negative_cttp.backward()
        #             pred_noise -= self.cond_configs["constraint"]["guide_w"] * self.generator.ddim.one_minus_alpha_bar_sqrt[t] * x0.grad.permute(0,2,1)
        #             x = self.generator.ddim.reverse(x, pred_noise, t, noise, is_determin=True)
        #     samples.append(x)
        # return torch.stack(samples)