



v0 / v2: supervised忘加时空编码了qwq
[DEBUG] x before embedding: torch.Size([12, 4, 28, 28])
[DEBUG] x after x_embedder: torch.Size([12, 196, 1152])
[DEBUG] pos_embed slice: torch.Size([1, 196, 1152])
[DEBUG] x after adding pos_embed: torch.Size([12, 196, 1152])



test:

export RESULTS_FOLDER=/data1/tpz/nwm-main/results
python isolated_nwm_infer_recon.py \
    --exp config/nwm_cdit_recon.yaml \
    --datasets recon \
    --batch_size 96 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER} \
    --gt 1



python isolated_nwm_infer_recon.py \
    --exp config/nwm_cdit_recon.yaml \
    --ckp 0100000 \
    --datasets recon \
    --batch_size 2 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}



python isolated_nwm_eval.py \
    --datasets recon \
    --gt_dir ${RESULTS_FOLDER}/gt \
    --exp_dir ${RESULTS_FOLDER}/nwm_cdit_recon \
    --eval_types time