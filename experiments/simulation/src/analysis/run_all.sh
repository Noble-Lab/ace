
echo "Obtaining results for each method on simulated clean and hard dataset."
'''
Rscript run_deseq2.R
Rscript run_grscore.R
Rscript run_jensen_shannon.R
cd GCE
python3 run_gce.py --input_data ../../../data/220var_clean.csv --tag 'clean'
python3 run_gce.py --input_data ../../../data/220var_hard.csv --tag 'hard'
cd ..
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_clean.csv --output_dir ../../results/clean --label clean --feat_dim 50 --beta 0.01 --plot_embedding 0 --margin 0.5 --latent_dim 10 --max_iter 2000
python ../explainer/tf_LRP_explainer.py --input_url ../../data/220var_hard.csv --output_dir ../../results/hard --label hard --feat_dim 75 --beta 1.0 --plot_embedding 0 --margin 0.5 --latent_dim 10 --batch_correct 2 --max_iter 2000
'''
echo "Calculating metrics for each method's gene ranking."

#python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/clean/one_vs_rest/deseq2_one_vs_rest.csv --method deseq2 --output_dir ../../results/metrics --type one_rest --tag clean
#python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/hard/one_vs_rest/deseq2_one_vs_rest.csv --method deseq2 --output_dir ../../results/metrics --type one_rest --tag hard

#python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/clean/one_vs_rest/grs_one_vs_rest.csv --method grs --output_dir ../../results/metrics --type one_rest --tag clean 
#python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/hard/one_vs_rest/grs_one_vs_rest.csv --method grs --output_dir ../../results/metrics --type one_rest --tag hard

#python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/clean/one_vs_rest/jsd_one_vs_rest.csv --method jsd --output_dir ../../results/metrics --type one_rest --tag clean 
#python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/hard/one_vs_rest/jsd_one_vs_rest.csv --method jsd --output_dir ../../results/metrics --type one_rest --tag hard

#python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/clean/one_vs_rest/gce_one_vs_rest.csv --method gce --output_dir ../../results/metrics --type one_rest --tag clean
#python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/hard/one_vs_rest/gce_one_vs_rest.csv --method gce --output_dir ../../results/metrics --type one_rest --tag hard

#python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/clean/one_vs_rest/ace_one_vs_rest_carlini_wagner_lamda100.0_iter2000_lr0.001_post1.csv --method ace --output_dir ../../results/metrics --type one_rest --tag clean
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/hard/one_vs_rest/ace_one_vs_rest_carlini_wagner_lamda100.0_iter2000_lr0.001_post1.csv --method ace --output_dir ../../results/metrics --type one_rest --tag hard

#python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/clean/one_vs_rest/ace_one_vs_rest_saliency_smoothgrad_post1.csv --method sal --output_dir ../../results/metrics --type one_rest --tag clean
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/hard/one_vs_rest/ace_one_vs_rest_saliency_smoothgrad_post1.csv --method sal --output_dir ../../results/metrics --type one_rest --tag hard

#python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/clean/one_vs_rest/ace_one_vs_rest_shap_post1.csv --method shap --output_dir ../../results/metrics --type one_rest --tag clean
python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/hard/one_vs_rest/ace_one_vs_rest_shap_post1.csv --method shap --output_dir ../../results/metrics --type one_rest --tag hard

# python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/clean/one_vs_rest/jsd_one_vs_rest.csv --method jsd --output_dir ../../results/metrics --type one_rest --tag clean