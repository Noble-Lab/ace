#!/bin/bash
# chmod +x execute.sh

echo "Getting results for each method on simulated dataset"
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_clean.csv --output_dir ../../results/method_output_220var --label 220var_clean --feat_dim -1 --beta 0.01 --plot_embedding 0 --margin 0 --latent_dim 2 --max_iter 2000
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_clean.csv --output_dir ../../results/method_output_220var --label 220var_clean --feat_dim -1 --beta 0.01 --plot_embedding 0 --margin 0.5 --latent_dim 2 --max_iter 2000
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_clean.csv --output_dir ../../results/method_output_220var --label 220var_clean --feat_dim -1 --beta 0.01 --plot_embedding 0 --margin 0 --latent_dim 10 --max_iter 2000
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_clean.csv --output_dir ../../results/method_output_220var --label 220var_clean --feat_dim -1 --beta 0.01 --plot_embedding 0 --margin 0.5 --latent_dim 10 --max_iter 2000
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_clean.csv --output_dir ../../results/method_output_220var --label 220var_clean --feat_dim 50 --beta 0.01 --plot_embedding 0 --margin 0 --latent_dim 2 --max_iter 2000
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_clean.csv --output_dir ../../results/method_output_220var --label 220var_clean --feat_dim 50 --beta 0.01 --plot_embedding 0 --margin 0.5 --latent_dim 2 --max_iter 2000
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_clean.csv --output_dir ../../results/method_output_220var --label 220var_clean --feat_dim 50 --beta 0.01 --plot_embedding 0 --margin 0 --latent_dim 10 --max_iter 2000
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_clean.csv --output_dir ../../results/method_output_220var --label 220var_clean --feat_dim 50 --beta 0.01 --plot_embedding 0 --margin 0.5 --latent_dim 10 --max_iter 2000

echo "Calculating metrics for each method"
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_clean_beta0.01_margin0_fd-1_latent2/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0_fd1_latent2
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_clean_beta0.01_margin0.5_fd-1_latent2/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0.5_fd1_latent2
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_clean_beta0.01_margin0_fd-1_latent10/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0_fd1_latent10
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_clean_beta0.01_margin0.5_fd-1_latent10/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0.5_fd1_latent10
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_clean_beta0.01_margin0_fd50_latent2/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0_fd50_latent2
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_clean_beta0.01_margin0.5_fd50_latent2/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0.5_fd50_latent2
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_clean_beta0.01_margin0_fd50_latent10/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0_fd50_latent10
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_clean_beta0.01_margin0.5_fd50_latent10/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0.5_fd50_latent10

echo "Getting results for each method on batch effects simulated dataset"
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_hard.csv --output_dir ../../results/method_output_220var --label 220var_hard --feat_dim -1 --beta 0.01 --plot_embedding 0 --margin 0 --latent_dim 2 --max_iter 2000 --batch_correct 1
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_hard.csv --output_dir ../../results/method_output_220var --label 220var_hard --feat_dim -1 --beta 0.01 --plot_embedding 0 --margin 0.5 --latent_dim 2 --max_iter 2000 --batch_correct 1
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_hard.csv --output_dir ../../results/method_output_220var --label 220var_hard --feat_dim -1 --beta 0.01 --plot_embedding 0 --margin 0 --latent_dim 10 --max_iter 2000 --batch_correct 1
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_hard.csv --output_dir ../../results/method_output_220var --label 220var_hard --feat_dim -1 --beta 0.01 --plot_embedding 0 --margin 0.5 --latent_dim 10 --max_iter 2000 --batch_correct 1
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_hard.csv --output_dir ../../results/method_output_220var --label 220var_hard --feat_dim 50 --beta 0.01 --plot_embedding 0 --margin 0 --latent_dim 2 --max_iter 2000 --batch_correct 1
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_hard.csv --output_dir ../../results/method_output_220var --label 220var_hard --feat_dim 50 --beta 0.01 --plot_embedding 0 --margin 0.5 --latent_dim 2 --max_iter 2000 --batch_correct 1
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_hard.csv --output_dir ../../results/method_output_220var --label 220var_hard --feat_dim 50 --beta 0.01 --plot_embedding 0 --margin 0 --latent_dim 10 --max_iter 2000 --batch_correct 1
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_hard.csv --output_dir ../../results/method_output_220var --label 220var_hard --feat_dim 50 --beta 0.01 --plot_embedding 0 --margin 0.5 --latent_dim 10 --max_iter 2000 --batch_correct 1

echo "Calculating metrics for each method with batch effects"
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_hard_beta0.01_margin0_fd-1_latent2/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0_fd1_latent2_hard
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_hard_beta0.01_margin0.5_fd-1_latent2/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0.5_fd1_latent2_hard
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_hard_beta0.01_margin0_fd-1_latent10/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0_fd1_latent10_hard
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_hard_beta0.01_margin0.5_fd-1_latent10/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0.5_fd1_latent10_hard
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_hard_beta0.01_margin0_fd50_latent2/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0_fd50_latent2_hard
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_hard_beta0.01_margin0.5_fd50_latent2/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0.5_fd50_latent2_hard
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_hard_beta0.01_margin0_fd50_latent10/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0_fd50_latent10_hard
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/method_output_220var/figs_220var_hard_beta0.01_margin0.5_fd50_latent10/explainer_result.csv --method explainer --output_dir ../../results/exp_metrics_220var --tag _mar0.5_fd50_latent10_hard

echo "done."