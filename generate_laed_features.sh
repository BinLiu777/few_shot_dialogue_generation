python generate_laed_features.py \
    logs/2021-06-07T10-35-18-st_ed.py \
    laed_features/st_ed_maluuba__smd_\${TARGET_DOMAIN} \
    --model_name StED \
    --model_type dialog \
    --data_dir NeuralDialog-ZSDG/data/stanford \
    --corpus_client ZslStanfordCorpus \
    --data_loader SMDDialogSkipLoader \
    --vocab maluuba.json  \
