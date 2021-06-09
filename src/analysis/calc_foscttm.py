import numpy as np
import random, math, os, sys, argparse
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
import csv
import random

# try summing over instead of min/max
# add explainer w/wo feature selection
# try w batch effects/ more dependencies
# submit cluster job
# want to try: 3 metrics x 5 baseline (gce, grscore, deseq2, jsd, random) + XXX combos () 

def calc_silhouette(mat, clust_labs):
	"""
	Given mat = (n x k matrix), and cluster labels for all n cells
	Calculate the mean silhouette score for all cells 
	"""
	score = silhouette_score(mat, clust_labs)
	return(score)

def calc_FOSCTTM(x1_mat, x2_mat):
	"""
	Calculate "fraction of samples closer than the true match"
	Given: X1 = (n x p matrix), and X2 = (n x k matrix)
	for every cell in X1, calculate cosine distance between itself and all other cells, and record its true nearest neighbor
	for every cell in X2, calculate cosine distance between itself and all other cells, sort and see where its true nearest neighbor was ranked
	record the fraction of cells that were closer than the true nearest neighbor
	"""
	nsamp = x1_mat.shape[0]
	x1_dist_mat = pairwise_distances(x1_mat, metric="cosine")
	x2_dist_mat = pairwise_distances(x2_mat, metric="cosine")

	# make all diagonal elts infinity
	np.fill_diagonal(x1_dist_mat, np.inf)
	np.fill_diagonal(x2_dist_mat, np.inf)

	fracs = []
	for i in range(0,nsamp):
		x1_samp_true_nbr = np.argmin(x1_dist_mat[i,:])
		sorted_samp_x2_dists = np.argsort(x2_dist_mat[i,:]).tolist()
		rank = sorted_samp_x2_dists.index(x1_samp_true_nbr)
		frac = float(rank)/(len(sorted_samp_x2_dists))
		fracs.append(frac)
	return np.mean(fracs)

def calc_jaccard_distance(x1_mat, x2_mat):
	"""
	X1 = (n x p matrix)
	X2 = (n x k matrix) where k < p
	for a cell in X1, find its k-nearest neighbors (set A)
	for a cell in X2, find its k-nearest neighbors (set B)
	jaccard distance = 1 - jaccard similarity = 1 - intersection(A,B)/union(A,B)
 	"""
	nsamp = x1_mat.shape[0]
	x1_nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(x1_mat)
	x2_nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(x2_mat)
	x1_nn = x1_nbrs.kneighbors(x1_mat, return_distance=False)
	x2_nn = x2_nbrs.kneighbors(x2_mat, return_distance=False)

	dists = []
	for i in range(0, nsamp):
		score = np.intersect1d(x1_nn[i,:], x2_nn[i,:]).shape[0]/np.union1d(x1_nn[i,:], x2_nn[i,:]).shape[0]
		jacc_dist = 1 - score
		dists.append(jacc_dist)
	return np.mean(dists)

def normalize_raw_data(raw_data):
	"""
	Normalizes input data of dimensions n x p 
	Log-transform with pseudocount
	"""
	raw = np.genfromtxt(raw_data, delimiter=',', skip_header=1, encoding='ascii');
	temp = np.asarray(raw[:, 1:], dtype=float);
	norm_data = np.log(temp + 1.0);

	cell_ids = np.genfromtxt(raw_data, delimiter=',', skip_header=1, usecols=(0), dtype=None, encoding='ascii');
	cell_ids = np.asarray(cell_ids, dtype=str);
	clust_ids = np.asarray([int(id.replace('"', '').split('_')[1]) for id in cell_ids], dtype=int);

	return norm_data, clust_ids

def make_idx_lookup(raw_data):
	"""
	makes a dictionary lookup table
	keys are genes and values are indices
	"""
	gene_ids = np.genfromtxt(args.raw_data, delimiter=',', max_rows=1, dtype=None);
	gene_ids = np.asarray(gene_ids[1:], dtype=str)

	idx_dict = {}
	for gene,idx in zip(gene_ids,range(0,len(gene_ids))):
		idx_dict[gene] = idx
	return idx_dict

def read_result(result, idx_lookup, method):
	"""
	Reads in the result table and sorts genes
	according to importance metric
	Returns the sorted gene indices
	"""
	sorted_idx = []
	if method == 'grscore':
		res = np.genfromtxt(result, delimiter=',', skip_header=1, dtype='unicode')
		sorted_res = res[res[:,1].argsort()[::-1]] # sort by global gene relevance
		sorted_genes = np.asarray(sorted_res[:,0])
		[sorted_idx.append(idx_lookup[gene]) for gene in sorted_genes]

	if method == 'deseq2':
		res = np.genfromtxt(result, delimiter=',', skip_header=1, dtype='unicode')
		sorted_res = res[np.asarray(res[:,1], dtype=float).argsort()] # sort by padj
		sorted_genes = np.asarray(sorted_res[:,0])
		[sorted_idx.append(idx_lookup[gene]) for gene in sorted_genes]

	if method == 'jensen-shannon':
		res = np.genfromtxt(result, delimiter=',', skip_header=1, dtype='unicode')
		sorted_res = res[np.asarray(res[:,1], dtype=float).argsort()] # sort by q-value
		sorted_genes = np.asarray(sorted_res[:,0])
		[sorted_idx.append(idx_lookup[gene]) for gene in sorted_genes]

	if method == 'gce':
		res = np.genfromtxt(result, delimiter=',', skip_header=0, dtype='unicode')
		sorted_idx = np.asarray(res, dtype=float).argsort()[::-1] # sort by explanation relevance

	if method == 'explainer':
		res = np.genfromtxt(result, delimiter=',', skip_header=0, dtype='unicode')
		sorted_idx = np.asarray(res, dtype=float).argsort()[::-1] # sort by explanation relevance 

	if method == 'random':
		genes = list(idx_lookup.values())
		sorted_idx = random.sample(genes, len(genes))

	return sorted_idx

def main(args):
	norm_data, clust_labels = normalize_raw_data(args.raw_data)
	idx_lookup_table = make_idx_lookup(args.raw_data)
	print("Dimensions of full data: ", norm_data.shape)

	sil = {}
	jd = {}
	foscttm = {}
	sorted_idx = read_result(args.result, idx_lookup_table, args.method)

	for i in range(1,norm_data.shape[1]+1):
		print("number of markers: ", str(i))
		if args.method == 'random':
			store_sil = []
			store_jaccard = []
			store_foscttm = []
			for j in range(10):
				random_rank = read_result(args.result, idx_lookup_table, args.method)
				store_sil.append(calc_silhouette(norm_data.T[random_rank][:i].T, clust_labels))
				store_jaccard.append(calc_jaccard_distance(norm_data, norm_data.T[random_rank][:i].T))
				store_foscttm.append(calc_FOSCTTM(norm_data, norm_data.T[random_rank][:i].T))
			sil[i] = np.mean(store_sil)
			jd[i] = np.mean(store_jaccard)
			foscttm[i] = np.mean(store_foscttm)
		else:
			sil_score = calc_silhouette(norm_data.T[sorted_idx][:i].T, clust_labels)
			dist = calc_jaccard_distance(norm_data, norm_data.T[sorted_idx][:i].T)
			frac = calc_FOSCTTM(norm_data, norm_data.T[sorted_idx][:i].T)
			sil[i] = sil_score
			jd[i] = dist
			foscttm[i] = frac
		print("mean silhouette score: ", str(sil[i]))
		print("mean jaccard distance: ", str(jd[i]))
		print("mean FOSCTTM: ", str(foscttm[i]))

	with open(os.path.join(args.output_dir, str(args.method + args.tag + '_sil.csv')), 'w') as csv_file:  
		writer = csv.writer(csv_file)
		for key, value in sil.items():
			writer.writerow([key, value])

	with open(os.path.join(args.output_dir, str(args.method + args.tag + '_jaccard.csv')), 'w') as csv_file:  
		writer = csv.writer(csv_file)
		for key, value in jd.items():
			writer.writerow([key, value])

	with open(os.path.join(args.output_dir, str(args.method + args.tag + '_foscttm.csv')), 'w') as csv_file:  
		writer = csv.writer(csv_file)
		for key, value in foscttm.items():
			writer.writerow([key, value])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--raw_data', type=str, help='raw data');
    parser.add_argument('--result', type=str, help='result');
    parser.add_argument('--method', type=str, help='method');
    parser.add_argument('--output_dir',type=str,help='output directory');
    parser.add_argument('--tag',type=str,help='tag',default='');

    args = parser.parse_args();
    main(args);	

### to run: python3 calc_foscttm.py --raw_data ../../data/trajectory_clean_counts.csv --result ../../results/morans_result.csv --method morans --output_dir ../../results
### to run: python3 calc_foscttm.py --raw_data ../../data/trajectory_clean_counts.csv --result ../../results/grscore_result.csv --method grscore --output_dir ../../results
### to run: python3 calc_foscttm.py --raw_data ../../data/130var_clean.csv --result ../../results/deseq2_result.csv --method deseq2 --output_dir ../../results



