
python run.py \
    --cond_modal multimodal \
    --text_type my_generated_text_embeds \
    --training_stage finetune \
    --save_folder ../save/synth_u_debug/qwen_my_embeds_v2 \
    --model_diff_config_path configs/synth_u_debug/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/synth_u_debug/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_u_debug/train.yaml \
    --evaluate_config_path configs/synth_u_debug/evaluate.yaml \
    --data_folder /Users/zhc/Documents/LitsDatasets/128_len_ts/synthetic_u \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 3 \
    --base_patch 4 \
    --epochs 700 \
    --batch_size 512 \
    --clip_cache_path ""

