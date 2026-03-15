import torch
import torch.nn as nn
from samplers import DDPMSampler, DDIMSampler
from models.diffusion.no_verbalts import NoVerbalTS



class Generator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.device = configs["device"]
        self.configs = configs
        self._init_diff(configs["diffusion"])

    def _init_diff(self, configs):
        configs["device"] = self.device
        self.diff_model = NoVerbalTS(configs, inputdim=1).to(self.device)

        # self.diff_model = DiTModel(configs).to(self.device)
        self.num_steps = configs["num_steps"]
        self.ddpm = DDPMSampler(self.num_steps, configs["beta_start"], configs["beta_end"], configs["schedule"],
                                self.device)
        self.ddim = DDIMSampler(self.num_steps, configs["beta_start"], configs["beta_end"], configs["schedule"],
                                self.device)

    def _noise_estimation_loss(self, x, tp, t):
        noise = torch.randn_like(x)
        noisy_x = self.ddpm.forward(x, t, noise)
        pred_noise, loss_dict = self.predict_noise(noisy_x, tp, t)
        residual = noise - pred_noise
        loss_dict["noise_loss"] = (residual ** 2).mean()
        all_loss = torch.zeros_like(loss_dict["noise_loss"])
        for k in loss_dict.keys():
            all_loss += loss_dict[k]
        loss_dict["all"] = all_loss
        return loss_dict

    def forward(self, batch, is_train):
        raise NotImplementedError

    def predict_noise(self, xt, tp, t):
        noisy_x = torch.unsqueeze(xt, 1)
        pred_noise, loss_dict = self.diff_model(noisy_x, tp, t)
        return pred_noise, loss_dict


class NoTextGenerator(nn.Module):
    def __init__(self, diff_configs, cond_configs):
        super().__init__()
        self.device = diff_configs["device"] if torch.cuda.is_available() else "cpu"
        self.diff_configs = diff_configs
        self.cond_configs = cond_configs
        self._init_diff(diff_configs)

    def _init_diff(self, configs):
        configs["device"] = self.device

        self.generator = Generator(configs=configs)
        if configs["generator_pretrain_path"] != "":
            self.generator.load_state_dict(torch.load(configs["generator_pretrain_path"]))
            print("Load the pretrain unconditional generator")
        else:
            print("Learn from scratch")

    def forward(self, batch, is_train):
        x, tp, _ = self._unpack_data_cond_gen_for_sample(batch)
        B, _, T = x.shape
        if is_train:
            t = torch.randint(0, self.generator.num_steps, [B], device=self.device)
            loss = self.generator._noise_estimation_loss(x, tp, t)
            return loss

        loss_dict = {}
        for t in range(self.generator.num_steps):
            t = (torch.ones(B, device=self.device) * t).long()

            tmp_loss_dict = self.generator._noise_estimation_loss(x, tp, t)
            for k in tmp_loss_dict:
                if k in loss_dict.keys():
                    loss_dict[k] += tmp_loss_dict[k]
                else:
                    loss_dict[k] = tmp_loss_dict[k]
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / self.generator.num_steps
        return loss_dict

    def _unpack_data_cond_gen_for_sample(self, batch):
        ts = batch["ts"].to(self.device).float()  # batch_size, num_channels, seq_len
        B, _, T = ts.shape
        tp = torch.arange(T).repeat(B, 1).to(self.device).float()
        text_embedding_all_segments = batch["text_embedding_all_segments"].to(self.device).float()
        # attn_mask = batch["attn_mask"].to(self.device).float()
        return ts, tp, text_embedding_all_segments

    def generate(self, batch, n_samples, sampler="ddim"):
        if self.cond_configs["cond_modal"] == "constraint":
            raise ValueError("Not Changed for precomputed attr_embed yet")
            # return self.generate_constraint(batch, n_samples, sampler)
        else:
            return self.generate_no_text(batch, n_samples, sampler)

    @torch.no_grad()
    def generate_no_text(self, batch, n_samples, sampler="ddim"):
        ts, tp, _ = self._unpack_data_cond_gen_for_sample(batch)
        samples = []
        B, _, T = ts.shape

        for i in range(n_samples):
            x = torch.randn_like(ts)

            for t in range(self.generator.num_steps - 1, -1, -1):
                noise = torch.randn_like(x)
                t = (torch.ones(B, device=self.device) * t).long()
                pred_noise, _ = self.generator.predict_noise(x, tp, t)
                if sampler == "ddpm":
                    x = self.generator.ddpm.reverse(x, pred_noise, t, noise)
                else:
                    x = self.generator.ddim.reverse(x, pred_noise, t, noise, is_determin=True)

            samples.append(x)
        return torch.stack(samples)
