import numpy as np
import os, sys, argparse
from sklearn.cluster import KMeans

sys.path.insert(0, "ELDR")
from base import load_encoder
from explain_cs import explain, apply
from metrics import metrics, eval_epsilon
from myplot import plot_groups, plot_metrics
from train_ae import train_ae
from sklearn import preprocessing

input_dim = 140
num_clusters = 2
ntrials = 5
lg = 1.5 # lambda_global

def load_model(input_dim = input_dim, model_file = "gce_model/model.cpkt"):
    return load_encoder(input_dim, model_file, encoder_shape = [20, 2])

def get_data(args):
	# cell ids
    cell_ids = np.genfromtxt(args.input_data, delimiter=',', skip_header=1, usecols=(0), dtype=None, encoding='ascii');
    cell_ids = np.asarray(cell_ids, dtype=str)
    # gene ids
    gene_ids = np.genfromtxt(args.input_data, delimiter=',', max_rows=1, dtype=None);
    gene_ids = np.asarray(gene_ids[1:], dtype=str)
    # cell profiles
    cell_profiles = np.genfromtxt(args.input_data, delimiter=',', skip_header=1, encoding='ascii');
    cell_profiles = np.asarray(cell_profiles[:, 1:], dtype=float);
    cell_profiles = np.log(cell_profiles + 1.0);
    cell_profiles = preprocessing.scale(cell_profiles);
    # cluster ids
    clust_ids = np.asarray([int(id.replace('"', '').split('_')[1]) for id in cell_ids], dtype=int);

    return cell_profiles, cell_ids, gene_ids, clust_ids;

def compute_explanation(cell_profiles, data_rep, clust_ids, clust_of_interest):
	# recode cluster ids
	new_ids = clust_ids.copy()
	new_ids[new_ids == clust_of_interest] = 0
	new_ids[new_ids != 0] = 1
	print(new_ids)

	# using actual cluster labels instead of kmeans labels
	means, centers, indices = plot_groups(cell_profiles, data_rep, num_clusters, new_ids, name = "gce_model/cluster" + str(clust_of_interest) + ".png")

	# evaluate epsilon. should yield metric values between 0.95-1
	epsilon = 1
	eval_epsilon(load_model, cell_profiles, indices, epsilon)

	# Compute the group explanations
	best_val = 0.0
	for i in range(ntrials):
		print("trial: " + str(i))
		deltas = explain(load_model, means, centers, learning_rate = 0.01, consecutive_steps = 5, lambda_global = lg)
		a, b = metrics(load_model, cell_profiles, indices, deltas, epsilon) 
		val = np.mean(a) 
		print(val)
		if val > best_val:
			best_val = val
			np.save("deltas" + str(clust_of_interest) + ".npy", deltas)
		os.system("rm -rf explanation")

	deltas = np.load("deltas" + str(clust_of_interest) + ".npy")

	# write out explanations
	exps_out = []
	for i in range(num_clusters - 1):
		exps_out.append(deltas[i])
	exps_out = np.array(exps_out)
	exps_out = np.fabs(exps_out)
	exps_out = np.amax(exps_out, axis=0)

	return exps_out

def main(args):
	cell_profiles, cell_ids, gene_ids, clust_ids = get_data(args);
	#train_ae(cell_profiles, encoder_shape = [20, 2], decoder_shape = [2, 20])
	sess, rep, X, D = load_model()
	data_rep = sess.run(rep, feed_dict={X: cell_profiles, D: np.zeros((1, input_dim))})
	#kmeans = KMeans(n_clusters = num_clusters).fit(data_rep)

	res = []
	for clust in np.unique(clust_ids):
		exp = compute_explanation(cell_profiles, data_rep, clust_ids, int(clust)).tolist()
		res.append(exp)
	print(res)
	max_rel = np.amax(res, axis=0)
	print(max_rel)
	np.savetxt("../../../results/method_output_140var/gce_result.csv", max_rel, delimiter=",", fmt = '%10.3f')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Optional app description');
	parser.add_argument('--input_data', type=str, help='input_data');
	args = parser.parse_args();
	main(args);

# python3 run_gce.py --input_data ../../../data/140var_clean.csv
