python generate_laed_features.py \
    logs/2021-06-08T17-35-33-ae_ed.py \
    laed_featuress/ae_ed \
    --model_name AeED \
    --model_type dialog \
    --data_dir NeuralDialog_ZSDG/data/stanford \
    --corpus_client ZslStanfordCorpusPre \
    --data_loader SMDDialogSkipLoader \
    --vocab maluuba.json  \
