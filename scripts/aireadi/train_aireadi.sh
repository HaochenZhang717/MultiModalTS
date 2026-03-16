export CUDA_VISIBLE_DEVICES=7
export USE_CAUSAL="true"


python run_forecast.py \
    --cond_modal aireadi \
    --text_type my_generated_text_embeds \
    --training_stage finetune \
    --save_folder ../save/aireadi/history_0315 \
    --samples_name "samples.pt" \
    --model_diff_config_path configs/aireadi/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/aireadi/cond/text_msmdiffmv.yaml \
    --train_config_path configs/aireadi/train.yaml \
    --evaluate_config_path configs/aireadi/evaluate.yaml \
    --data_folder "none" \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 700 \
    --batch_size 512 \
    --clip_cache_path "" \
    --condition_type "history"


python run_forecast.py \
    --cond_modal aireadi \
    --text_type my_generated_text_embeds \
    --training_stage finetune \
    --save_folder ../save/aireadi/retinal_text_history_0315 \
    --samples_name "samples.pt" \
    --model_diff_config_path configs/aireadi/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/aireadi/cond/text_msmdiffmv.yaml \
    --train_config_path configs/aireadi/train.yaml \
    --evaluate_config_path configs/aireadi/evaluate.yaml \
    --data_folder "none" \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 700 \
    --batch_size 512 \
    --clip_cache_path "" \
    --condition_type "adaLN"




