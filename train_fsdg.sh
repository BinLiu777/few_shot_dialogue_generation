CUDA_VISIBLE_DEVICES=2 python train_fsdg.py \
    LAZslStanfordCorpus \
    --data_dir NeuralDialog_ZSDG/data/stanford \
    --laed_z_folders laed_features/ae_ed laed_features/st_ed \
    --black_domains \$domain \
    --black_ratio 0.9 \
    --action_match False \
    --target_example_cnt 0 \
    --random_seed \$rnd