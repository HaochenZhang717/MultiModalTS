export HF_HOME=/playpen/haochenz/hf_cache

CUDA_VISIBLE_DEVICES=6 python run_causal.py \
    --cond_modal multimodal \
    --text_type my_generated_text_embeds \
    --training_stage finetune \
    --save_folder ../save/synth_u_causal/qwen_my_embeds_v2_0312 \
    --model_diff_config_path configs/synth_u_causal/diff/model.yaml \
    --model_cond_config_path configs/synth_u_causal/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_u_causal/train.yaml \
    --evaluate_config_path configs/synth_u_causal/evaluate.yaml \
    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 3 \
    --base_patch 4 \
    --epochs 700 \
    --batch_size 512 \
    --clip_cache_path ""
