export HF_HOME=/playpen/haochenz/hf_cache
# todo: need to change
CUDA_VISIBLE_DEVICES=6 python sample_only.py \
    --cond_modal multimodal \
    --text_type my_generated_text_embeds \
    --training_stage finetune \
    --save_folder ../save/synth_u/use_qwen_my_embedding \
    --model_diff_config_path configs/synth_u/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/synth_u/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_u/train.yaml \
    --evaluate_config_path configs/synth_u/evaluate.yaml \
    --data_folder /playpen/haochenz/synthetic_u \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 3 \
    --base_patch 4 \
    --epochs 700 \
    --batch_size 512 \
    --clip_cache_path "" \
    --text_embeds_path /playpen/haochenz/diffusion_prior_results/DiTDH-S-samples_embed.pt \
    --seq_len 128