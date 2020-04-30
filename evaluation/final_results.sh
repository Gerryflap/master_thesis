#!/bin/sh


# This file runs a number of final evaluations on important models

# Go to the root project directory
cd ..


# ===================  Experiments ======================
# MorGAN
echo "MorGAN"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/validation_results/MorGAN_normal/2019-12-18T22:19:30/params/epoch_000120/ --eval --batch_size=32 --visualize --test

# MorGAN+
echo "MorGAN+"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/validation_results/MorGANp_normal/2020-01-16T11:47:38/params/all_epochs/ --eval --batch_size=32 --visualize --test

# MorGAN+ dis_l 0.3
echo "MorGAN+ dis_l 0.3"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/validation_results/MorGANp_disl/2020-01-15T12:37:44/params/all_epochs/ --eval --batch_size=32 --visualize --test

# MorGAN+ 3.0 (MorGAN/2020-01-21T15:31:40)
echo "MorGAN+ 3.0 (MorGAN/2020-01-21T15:31:40)"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/results/MorGAN/2020-01-21T15:31:40/params/all_epochs/ --eval --batch_size=32 --visualize --test

# MorGAN+ dis_l 3.0
echo "MorGAN+ dis_l 3.0"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/validation_results/MorGANp_disl_3/2020-01-19T11:10:36/params/all_epochs/ --eval --batch_size=32 --visualize --test

# MorGAN+ dis_l 30.0
echo "MorGAN+ dis_l 30.0"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/validation_results/MorGANp_disl_30/2020-01-22T14:35:20/params/all_epochs/ --eval --batch_size=32 --visualize --test

# MorGAN+ dis_l 3.0 3.0 (with morph loss)
echo "MorGAN+ dis_l 3.0 3.0 (with morph loss)"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/results/celeba64/Morphing_GAN/2020-03-23T17:21:15/params/all_epochs/ --eval --batch_size=32 --visualize --test

# Morph network (Morphing_GAN/2020-04-17T18:29:45)
echo "Morph network (Morphing_GAN/2020-04-17T18:29:45)"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/results/celeba64/Morphing_GAN/2020-04-17T18:29:45/params/all_epochs/ --eval --batch_size=32 --visualize --test

# Morph network linear interpolation (so without actually using the morph network) (Morphing_GAN/2020-04-17T18:29:45)
echo "Morph network linear interpolation (so without actually using the morph network) (Morphing_GAN/2020-04-17T18:29:45)"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/rresults/celeba64/Morphing_GAN/2020-04-17T18:29:45/params/all_epochs/ --eval --batch_size=32 --visualize --test --force_linear_morph

# MorGAN+ dis_l 3.0 3.0 (with morph loss) (no Gx loss)
echo "MorGAN+ dis_l 3.0 3.0 (with morph loss) (no Gx loss)"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/results/celeba64/Morphing_GAN/2020-03-17T15:18:13/params/all_epochs/ --eval --batch_size=32 --visualize --test

# MorGAN+ dis_l 3.0 3.0 (with morph loss) (no Gz loss)
echo "MorGAN+ dis_l 3.0 3.0 (with morph loss) (no Gz loss)"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/results/celeba64/Morphing_GAN/2020-03-22T05:04:11/params/all_epochs/ --eval --batch_size=32 --visualize --test

# MorGAN+ dis_l 0.3 optim z
echo "MorGAN+ dis_l 0.3 optim z"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/validation_results/MorGANp_disl/2020-01-15T12:37:44/params/all_epochs/ --eval --batch_size=32 --visualize --test --gradient_descend_dis_l --gradient_descend_dis_l_recon

# MorGAN+ dis_l 3.0 optim z
echo "MorGAN+ dis_l 3.0 optim z"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/validation_results/MorGANp_disl_3/2020-01-19T11:10:36/params/all_epochs/ --eval --batch_size=32 --visualize --test --gradient_descend_dis_l --gradient_descend_dis_l_recon

# MorGAN+ dis_l 30.0
echo "MorGAN+ dis_l 30.0 optim z"
python -m evaluation.mmpmr_evaluator --cuda --parameter_path=hdd_store/validation_results/MorGANp_disl_30/2020-01-22T14:35:20/params/all_epochs/ --eval --batch_size=32 --visualize --test --gradient_descend_dis_l --gradient_descend_dis_l_recon