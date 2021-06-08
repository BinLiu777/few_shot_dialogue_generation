python generate_vae_features.py \
    logs/2021-06-06T17-57-32-vae.py \
    vae_features/navigate \
    --model_name AeED \
    --model_type dialog \
    --data_dir NeuralDialog_ZSDG/data/stanford \
    --corpus_client ZslStanfordCorpus \
    --data_loader SMDDialogSkipLoader \
    --vocab vocabs/maluuba.json
