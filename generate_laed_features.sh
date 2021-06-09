python generate_laed_features.py \
    logs/2021-06-08T17-10-45-ae_ed.py \
    laed_features/ae_ed \
    --model_name AeED \
    --model_type dialog \
    --data_dir NeuralDialog_ZSDG/data/customer_service \
    --corpus_client ZslStanfordCorpus \
    --data_loader SMDDialogSkipLoader \
    --vocab maluuba.json  \
