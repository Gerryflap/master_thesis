#!/bin/sh


# This file runs all gradient descend evaluations again.
# This was done due to an error in the code

# Go to the root project directory
cd ..

# MorGAN+ dis_l 0.3 optim z
echo "MorGAN+ dis_l 0.3 optim z"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/validation_results/MorGANp_disl/2020-01-15T12:37:44/params/all_epochs/ --eval --batch_size=32 --visualize --test --use_z_mean --gradient_descend_dis_l --gradient_descend_dis_l_recon

# MorGAN+ dis_l 3.0 optim z
echo "MorGAN+ dis_l 3.0 optim z"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/validation_results/MorGANp_disl_3/2020-01-19T11:10:36/params/all_epochs/ --eval --batch_size=32 --visualize --test --use_z_mean --gradient_descend_dis_l --gradient_descend_dis_l_recon

# MorGAN+ dis_l 30.0
echo "MorGAN+ dis_l 30.0 optim z"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/validation_results/MorGANp_disl_30/2020-01-22T14:35:20/params/all_epochs/ --eval --batch_size=32 --visualize --test --use_z_mean --gradient_descend_dis_l --gradient_descend_dis_l_recon