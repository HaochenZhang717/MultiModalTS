export HF_HOME=/playpen/haochenz/hf_cache
export USE_CAUSAL="true"


CUDA_VISIBLE_DEVICES=7 python run_causal.py \
    --cond_modal multimodal \
    --text_type my_generated_text_embeds \
    --training_stage finetune \
    --save_folder ../save/causal_correct/synth_m \
    --samples_name "samples.pt" \
    --model_diff_config_path configs/synth_m_causal/diff/model.yaml \
    --model_cond_config_path configs/synth_m_causal/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_m_causal/train.yaml \
    --evaluate_config_path configs/synth_m_causal/evaluate.yaml \
    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_m \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 2500 \
    --batch_size 512 \
    --clip_cache_path "" \
    --n_runs 1

CUDA_VISIBLE_DEVICES=7 python run_causal.py \
    --cond_modal multimodal \
    --text_type my_generated_text_embeds \
    --training_stage finetune \
    --save_folder ../save/causal_correct/istanbul_traffic \
    --samples_name "samples.pt" \
    --model_diff_config_path configs/istanbul_traffic_causal/diff/model.yaml \
    --model_cond_config_path configs/istanbul_traffic_causal/cond/text_msmdiffmv.yaml \
    --train_config_path configs/istanbul_traffic_causal/train.yaml \
    --evaluate_config_path configs/istanbul_traffic_causal/evaluate.yaml \
    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/istanbul_traffic \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 2500 \
    --batch_size 512 \
    --clip_cache_path "" \
    --n_runs 1


CUDA_VISIBLE_DEVICES=7 python run_causal.py \
    --cond_modal multimodal \
    --text_type my_generated_text_embeds \
    --training_stage finetune \
    --save_folder ../save/causal_correct/synth_u \
    --samples_name "samples.pt" \
    --model_diff_config_path configs/synth_u_causal/diff/model.yaml \
    --model_cond_config_path configs/synth_u_causal/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_u_causal/train.yaml \
    --evaluate_config_path configs/synth_u_causal/evaluate.yaml \
    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 2500 \
    --batch_size 512 \
    --clip_cache_path "" \
    --n_runs 1