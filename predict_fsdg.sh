python predict_fsdg.py \
    LAZslStanfordCorpus \
    --data_dir NeuralDialog_ZSDG/data/stanford\
    --laed_z_folders laed_features/st_ed \
    --black_domains \
    --black_ratio 0.9 \
    --action_match False \
    --target_example_cnt 0 \
    --random_seed 2021 \
    --forward_only 1 \
    --use_gpu 1 \
    --load_sess 2021-06-09T19-31-12-train_fsdg.py-6a61fab0