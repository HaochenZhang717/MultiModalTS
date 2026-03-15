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





class ConditionalPredictor(nn.Module):
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
        x, tp, text_embedding_all_segments = self._unpack_data_cond_gen(batch)
        B, _, T = x.shape

        BLOCK_ID = torch.randint(0, 4, (1,)).item()
        PREDICT_START = BLOCK_ID * 32
        PREDICT_END = BLOCK_ID * 32 + 32


        # here we test only predict the first block
        attn_mask = torch.zeros((B, T)).to(x.device)  # (B ,T)
        attn_mask[:, :PREDICT_END] = 1

        loss_mask = torch.zeros((B, T)).to(x.device)  # (B ,T)
        loss_mask[:, PREDICT_START:PREDICT_END] = 1

        text_embed = text_embedding_all_segments[:, BLOCK_ID]

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
        ts = batch["ts"].to(self.device).float()  # batch_size, num_channels, seq_len
        B, _, T = ts.shape
        tp = torch.arange(T).repeat(B, 1).to(self.device).float()
        text_embedding_all_segments = batch["text_embedding_all_segments"].to(self.device).float()
        return ts, tp, text_embedding_all_segments

    def generate(self, batch, n_samples, sampler="ddim"):
        if self.cond_configs["cond_modal"] == "constraint":
            raise ValueError("Not Changed for precomputed attr_embed yet")
            # return self.generate_constraint(batch, n_samples, sampler)
        else:
            return self.generate_text(batch, n_samples, sampler)


    # @torch.no_grad()
    # def generate_text(self, batch, n_samples, sampler="ddim"):
    #
    #     # BLOCK_ID = torch.randint(0, 3, (1,)).item()
    #     # BLOCK_ID = 3
    #     # PREDICT_START = BLOCK_ID * 32
    #     # PREDICT_END = BLOCK_ID * 32 + 32
    #
    #     ts, tp, text_embed_all_segments = self._unpack_data_cond_gen_for_sample(batch)
    #
    #     samples = []
    #     B, _, T = ts.shape
    #     # here we test only predict the first block
    #
    #
    #
    #     for i in range(n_samples):
    #         batch_samples = torch.zeros_like(ts)
    #         for block_id in range(4):
    #             predict_start = block_id * 32
    #             predict_end = block_id * 32 + 32
    #             attn_mask = torch.zeros((B, T)).to(ts.device)  # (B ,T)
    #             attn_mask[:, :predict_end] = 1
    #
    #             x = torch.randn_like(ts)
    #             x[:,:,:predict_start] = batch_samples[:,:,:predict_start]
    #             text_embed = text_embed_all_segments[:, block_id]
    #             attr_emb = self.cond_projector(text_embed)
    #
    #             for t in range(self.generator.num_steps-1, -1, -1):
    #                 noise = torch.randn_like(x)
    #                 t = (torch.ones(B, device=self.device) * t).long()
    #                 pred_noise, _ = self.generator.predict_noise(x, tp, attr_emb, t, attn_mask=attn_mask)
    #                 if sampler == "ddpm":
    #                     x = self.generator.ddpm.reverse(x, pred_noise, t, noise)
    #                 else:
    #                     x = self.generator.ddim.reverse(x, pred_noise, t, noise, is_determin=True)
    #
    #                 x[:, :, :predict_start] = batch_samples[:, :, :predict_start]
    #
    #             batch_samples[:, :, predict_start:predict_end] = x[:, :, predict_start:predict_end]
    #
    #         samples.append(batch_samples)
    #     return torch.stack(samples)



    @torch.no_grad()
    def generate_text(self, batch, n_samples, sampler="ddim"):

        # BLOCK_ID = torch.randint(0, 3, (1,)).item()
        BLOCK_ID = 0
        PREDICT_START = BLOCK_ID * 32
        PREDICT_END = BLOCK_ID * 32 + 32

        ts, tp, text_embed_all_segments = self._unpack_data_cond_gen(batch)
        text_embed = text_embed_all_segments[:, BLOCK_ID]
        samples = []
        B, _, T = ts.shape
        # here we test only predict the first block
        attn_mask = torch.zeros((B, T)).to(ts.device)  # (B ,T)
        attn_mask[:, :PREDICT_END] = 1

        for i in range(n_samples):
            x = torch.randn_like(ts)
            x[:,:,:PREDICT_START] = ts[:,:,:PREDICT_START]
            attr_emb = self.cond_projector(text_embed)

            for t in range(self.generator.num_steps-1, -1, -1):
                noise = torch.randn_like(x)
                t = (torch.ones(B, device=self.device) * t).long()
                pred_noise, _ = self.generator.predict_noise(x, tp, attr_emb, t, attn_mask=attn_mask)
                if sampler == "ddpm":
                    x = self.generator.ddpm.reverse(x, pred_noise, t, noise)
                else:
                    x = self.generator.ddim.reverse(x, pred_noise, t, noise, is_determin=True)

                x[:, :, :PREDICT_START] = ts[:, :, :PREDICT_START]

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