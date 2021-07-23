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

input_dim = 1838
ntrials = 1
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

def get_PBMC_data():
	cell_profiles = np.genfromtxt('../../data/3K_PBMC_matrix.txt', delimiter=',')
	gene_ids = np.genfromtxt('../../data/3K_PBMC_gene_ids.txt', dtype=None, encoding=None);
	cell_ids = np.genfromtxt('../../data/3K_PBMC_cell_ids.txt', dtype=None, encoding=None);
	clust_ids = np.genfromtxt('../../data/3K_PBMC_clust_ids_leiden.txt').astype(int)+1;
	return cell_profiles, cell_ids, gene_ids, clust_ids;

def gce_one_vs_rest(args, cell_profiles, data_rep, clust_ids, gene_ids, clust_of_interest, num_clusters):
	# recode cluster ids
	new_ids = clust_ids.copy()
	new_ids[new_ids == clust_of_interest] = 0
	new_ids[new_ids != 0] = 1

	# using actual cluster labels instead of kmeans labels
	means, centers, indices = plot_groups(cell_profiles, data_rep, num_clusters, new_ids, name = "gce_one_vs_rest_leiden/cluster" + str(clust_of_interest-1) + ".png")

	# evaluate epsilon. should yield metric values between 0.95-1
	epsilon = 1;
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
			np.save("gce_one_vs_rest_leiden/deltas" + str(clust_of_interest-1) + ".npy", deltas)
		os.system("rm -rf explanation")

	deltas = np.load("gce_one_vs_rest_leiden/deltas" + str(clust_of_interest-1) + ".npy")
	clust_vec = np.full((len(gene_ids)), clust_of_interest-1);
	exps_out = np.stack([clust_vec, gene_ids, np.fabs(deltas).reshape(-1)], axis=1);

	return exps_out

def main(args):
	cell_profiles, cell_ids, gene_ids, clust_ids = get_PBMC_data();
	#train_ae(cell_profiles, encoder_shape = [20, 2], decoder_shape = [2, 20])
	sess, rep, X, D = load_model()
	data_rep = sess.run(rep, feed_dict={X: cell_profiles, D: np.zeros((1, input_dim))})

	# one vs. rest
	num_clusters = 2
	res = np.empty((0,3), int);
	for clust in np.unique(clust_ids):
		exp = gce_one_vs_rest(args, cell_profiles, data_rep, clust_ids, gene_ids, int(clust), num_clusters).tolist()
		res = np.append(res, exp, axis=0);
	np.savetxt("../../results/gce_leiden.csv", res, delimiter=",", fmt = '%s')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Optional app description');
	parser.add_argument('--input_data', type=str, help='input_data');
	parser.add_argument('--tag', type=str, help='tag')
	args = parser.parse_args();
	main(args);
