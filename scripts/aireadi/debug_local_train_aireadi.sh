
python run.py \
    --cond_modal aireadi \
    --text_type my_generated_text_embeds \
    --training_stage finetune \
    --save_folder ../save/aireadi_debug/qwen_my_embeds_v2 \
    --model_diff_config_path configs/aireadi_debug/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/aireadi_debug/cond/text_msmdiffmv.yaml \
    --train_config_path configs/aireadi_debug/train.yaml \
    --evaluate_config_path configs/aireadi_debug/evaluate.yaml \
    --data_folder /Users/zhc/Documents/PhD/projects/TimeSeriesUnifiedModel/VerbalTSDatasets/synthetic_u \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 3 \
    --base_patch 4 \
    --epochs 700 \
    --batch_size 512 \
    --clip_cache_path ""

