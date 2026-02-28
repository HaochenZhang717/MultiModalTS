import os
import yaml
import json
import datetime
import argparse
import pandas as pd
import torch
import numpy as np
import random

from data import GenerationDataset
from models.conditional_generator import ConditionalGenerator
from models.unconditional_generator import UnConditionalGenerator
from train.trainer import Trainer
from evaluation.base_evaluator import BaseEvaluator
from time import time


def make_dummy_batch(text_embed_batch, n_steps, n_attrs):
    """
    text_embed_batch: torch.Tensor [B, D]

    Returns:
        dict with same structure as CustomSplit.__getitem__ batched
    """

    B = text_embed_batch.shape[0]
    device = text_embed_batch.device

    # Dummy time series (not used)
    ts = torch.zeros(B, n_steps, 1, device=device)

    batch = {
        "ts": ts,                              # [B, T, 1]
        "ts_len": torch.full((B,), n_steps),   # [B]
        "attrs": torch.zeros(B, n_attrs, device=device),
        "cap": ["dummy caption"] * B,                       # dummy text
        "tp": torch.arange(n_steps).unsqueeze(0).repeat(B, 1),
        "cap_embed": text_embed_batch          # [B, D]
    }

    return batch


def save_configs(configs, path):
    print(json.dumps(configs, indent=4))
    with open(path, "w") as f:
        yaml.dump(configs, f, yaml.SafeDumper)


def _cond_gen(model, text_embeds, n_steps, batch_size, device, mode="cond_gen", sampler="ddpm"):
    print("\n-------------------------------")
    print(f"Evaluating the model with mode={mode} and sampler={sampler}")
    model.eval().to(device)
    text_embeds = text_embeds.to(device)
    n_samples = 10
    dataset = torch.utils.data.TensorDataset(text_embeds.to(device))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    sampled_ts = []
    with torch.no_grad():
        for batch_no, text_embed_batch in enumerate(dataloader):
            batch = make_dummy_batch(text_embed_batch[0], n_steps, n_attrs=1)
            start_time = time()
            multi_preds = model.generate(batch, n_samples, sampler)
            multi_preds = multi_preds.permute(0, 1, 3, 2)
            end_time = time()
            print(f"Batch no={batch_no}, time={end_time - start_time}")
            pred = multi_preds.median(dim=0).values

            sampled_ts.append(multi_preds.cpu())

    sampled_ts = torch.cat(sampled_ts, dim=1)

    return sampled_ts


def evaluate(seq_len, text_embeds, eval_configs, model_diff_configs, model_cond_configs, output_folder):
    eval_configs["eval"]["model_path"] = os.path.join(output_folder, "ckpts/model_best_loss.pth")

    if "attrs" in model_cond_configs.keys():
        raise NotImplementedError
        # model_cond_configs["attrs"]["num_attr_ops"] = dataset.num_attr_ops.tolist()
    model = ConditionalGenerator(model_diff_configs, model_cond_configs)

    sampled_ts = _cond_gen(
        model, text_embeds,
        batch_size=512,
        n_steps=seq_len,
        device=model_diff_configs["device"],
        mode="cond_gen",
        sampler="ddpm"
    )
    torch.save(sampled_ts, os.path.join(output_folder, "sampled_ts.pth"))
    print("Saved sampled time series to {}".format(os.path.join(output_folder, "sampled_ts.pth")))
    return sampled_ts


def _evaluate_cond_gen(evaluator, output_folder, sampler="ddim", n_sample=10):
    evaluator.n_samples = n_sample
    res_dict, result_ts_dict = evaluator.evaluate(mode="cond_gen", sampler=sampler, save_pred=False)
    torch.save(result_ts_dict, os.path.join(output_folder, "samples.pth"))
    print("Samples saved in {}".format(os.path.join(output_folder, "samples.pth")))
    info = {
        "mode": "cond_gen",
        "sampler": sampler,
        "n_samples": evaluator.n_samples,
        "steps": -1,
    }
    info.update(res_dict["df"])
    df = pd.DataFrame([info])
    df["steps"].astype(int)
    return df


def run(seq_len, text_embeds, eval_configs, model_diff_configs, model_cond_configs, output_folder, data_folder=""):
    eval_configs["data"]["folder"] = data_folder

    sampled_ts = evaluate(seq_len, text_embeds, eval_configs, model_diff_configs, model_cond_configs, output_folder)

    return sampled_ts
##### Arguments #####
parser = argparse.ArgumentParser(description="TSE")
parser.add_argument("--training_stage", type=str, default="pretrain")
parser.add_argument("--model_diff_config_path", type=str, default="")
parser.add_argument("--model_cond_config_path", type=str, default="")
parser.add_argument("--generator_pretrain_path", type=str, default="")
parser.add_argument("--train_config_path", type=str, default="")
parser.add_argument("--evaluate_config_path", type=str, default="")
parser.add_argument("--data_folder", type=str, default="./datasets")
parser.add_argument("--save_folder", type=str, default="./save")
parser.add_argument("--clip_folder", type=str, default="")
parser.add_argument("--start_runid", type=int, default=0)
parser.add_argument("--n_runs", type=int, default=3)
parser.add_argument("--clip_cache_path", type=str, default="cache")

parser.add_argument("--cond_modal", type=str, default="text")
parser.add_argument("--text_output_type", type=str, default="all")
parser.add_argument("--text_pos_emb", type=str, default="none")

parser.add_argument("--base_patch", type=int, default=1)
parser.add_argument("--multipatch_num", type=int, default=3)
parser.add_argument("--L_patch_len", type=int, default=3)
parser.add_argument("--diff_stage_num", type=int, default=3)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=200)

parser.add_argument("--guide_w", type=float, default=1.0)
parser.add_argument("--only_evaluate", type=bool, default=False)


# extra parameters
parser.add_argument("--text_embeds_path", type=str, required=True)
parser.add_argument("--seq_len", type=int, required=True)
parser.add_argument("--text_type", type=str, required=True,
                    choices=["original_text_caps_only", "original_text_embeds", "my_generated_text", "my_generated_text_embeds"]
                    )



args = parser.parse_args()

save_folder = args.save_folder
os.makedirs(save_folder, exist_ok=True)
print("All files will be saved to '{}'".format(save_folder))

train_configs = yaml.safe_load(open(args.train_config_path))
eval_configs = yaml.safe_load(open(args.evaluate_config_path))

train_configs["train"].update({"text_type": args.text_type})
eval_configs["eval"].update({"text_type": args.text_type})

model_diff_configs = yaml.safe_load(open(args.model_diff_config_path))
if args.training_stage == "finetune":
    model_cond_configs = yaml.safe_load(open(args.model_cond_config_path))
    model_cond_configs["cond_modal"] = args.cond_modal
else:
    model_cond_configs = None

train_configs["data"]["folder"] = fr"{args.data_folder}"
train_configs["train"]["lr"] = args.lr
train_configs["train"]["epochs"] = args.epochs
train_configs["train"]["batch_size"] = args.batch_size
eval_configs["eval"]["batch_size"] = args.batch_size

model_diff_configs["diffusion"]["multipatch_num"] = args.multipatch_num
model_diff_configs["diffusion"]["L_patch_len"] = args.L_patch_len
model_diff_configs["diffusion"]["base_patch"] = args.base_patch
if "text" in args.model_cond_config_path and args.training_stage == "finetune":
    model_cond_configs["text"]["output_type"] = args.text_output_type
    model_cond_configs["text"]["num_stages"] = args.diff_stage_num
    model_cond_configs["text"]["pos_emb"] = args.text_pos_emb

if args.clip_folder != "":
    eval_configs["eval"]["cache_folder"] = args.clip_cache_path
    eval_configs["eval"]["clip_model_path"] = fr"{args.clip_folder}/clip_model_best.pth"
    eval_configs["eval"]["clip_config_path"] = fr"{args.clip_folder}/model_configs.yaml"
    
    if model_cond_configs["cond_modal"] == "constraint":
        model_cond_configs["constraint"]["clip_config_path"] = fr"{args.clip_folder}/clip_model_best.pth"
        model_cond_configs["constraint"]["clip_model_path"] = fr"{args.clip_folder}/model_configs.yaml"
        model_cond_configs["constraint"]["guide_w"] = args.guide_w

seed_list = [1, 7, 42]
df_list = []

eval_record_folder = eval_configs["data"]["folder"]

fix_seed = seed_list[0]
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

print(f"\nRun:")
output_folder = os.path.join(save_folder, str(0))
os.makedirs(output_folder, exist_ok=True)
eval_configs["eval"]["model_path"] = ""
eval_configs["data"]["folder"] = eval_record_folder
if args.generator_pretrain_path != "":
    model_diff_configs["generator_pretrain_path"] = f"{args.generator_pretrain_path}/{n}/ckpts/model_best_loss.pth"
else:
    model_diff_configs["generator_pretrain_path"] = ""


# text_embeds = np.load(args.text_embeds_path, allow_pickle=True)
text_embeds = torch.load(args.text_embeds_path, weights_only=False, map_location="cpu")
run(args.seq_len, text_embeds, eval_configs, model_diff_configs, model_cond_configs, output_folder, data_folder=args.data_folder)
