import numpy as np
import random, math, os, sys, argparse
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import csv
import random

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

def calc_jaccard_distance(x1_mat, x2_mat, cell_indices):
	"""
	X1 = (n x p matrix)
	X2 = (n x k matrix) where k < p
	for a cell in X1, find its k-nearest neighbors (set A)
	for a cell in X2, find its k-nearest neighbors (set B)
	jaccard distance = 1 - jaccard similarity = 1 - intersection(A,B)/union(A,B)
 	"""
	nsamp = x1_mat.shape[0]
	x1_nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(x1_mat)
	x2_nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(x2_mat)
	x1_nn = x1_nbrs.kneighbors(x1_mat, return_distance=False)
	x2_nn = x2_nbrs.kneighbors(x2_mat, return_distance=False)

	dists = []
	for i in cell_indices:
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
	norm_data = preprocessing.scale(norm_data);

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

def sort_by_relevance_one_vs_rest(res, method, idx_lookup):
	'''
	takes a cluster specific results object
	returns the sorted gene indices for the ranking
	'''
	sorted_idx = []
	if method in ['deseq2', 'jsd']:
		sorted_res = res[np.asarray(res[:,2], dtype=float).argsort()] # sort by the p-value
		sorted_genes = np.asarray(sorted_res[:,1])
		[sorted_idx.append(idx_lookup[gene]) for gene in sorted_genes]

	if method in ['ace', 'gce', 'grs', 'sal', 'shap']:
		sorted_res = res[res[:,2].argsort()[::-1]] # sort by gene relevance difference or explanation relevance
		sorted_genes = np.asarray(sorted_res[:,1])
		[sorted_idx.append(idx_lookup[gene]) for gene in sorted_genes]

	#if method == 'gce':
	#	sorted_idx = np.asarray(res, dtype=float).argsort()[::-1] # sort by explanation relevance
	return sorted_idx

def sort_by_relevance_one_vs_one(res, method, idx_lookup):
	'''
	takes a cluster-pair specific results object
	returns the sorted gene indices for the ranking
	'''
	sorted_idx = []
	if method in ['deseq2', 'jsd']:
		sorted_res = res[np.asarray(res[:,3], dtype=float).argsort()] # sort by the p-value
		sorted_genes = np.asarray(sorted_res[:,2])
		[sorted_idx.append(idx_lookup[gene]) for gene in sorted_genes]

	if method in ['ace', 'gce', 'grs', 'sal', 'shap']:
		sorted_res = res[res[:,3].argsort()[::-1]] # sort by gene relevance difference or explanation relevance
		sorted_genes = np.asarray(sorted_res[:,2])
		[sorted_idx.append(idx_lookup[gene]) for gene in sorted_genes]

	#if method == 'gce':
	#	sorted_idx = np.asarray(res[:,3], dtype=float).argsort()[::-1] # sort by explanation relevance
	return sorted_idx

def obtain_ranking_one_vs_rest(res, method, cluster, idx_lookup):
	"""
	reads in the full result table
	returns the gene ranking for a specific cluster
	"""
	clust_res = res[np.char.strip(res[:, 0], '"').astype(int) == cluster]
	ranking = sort_by_relevance_one_vs_rest(clust_res, method, idx_lookup)
	return ranking

def obtain_ranking_one_vs_one(res, method, cluster1, cluster2, idx_lookup):	
	"""
	reads in the full result table
	returns the gene ranking for a specific cluster-pair 
	"""
	res = res[np.char.strip(res[:, 0], '"').astype(int) == cluster1]
	clust_res = res[np.char.strip(res[:, 1], '"').astype(int) == cluster2]
	ranking = sort_by_relevance_one_vs_one(clust_res, method, idx_lookup)
	return ranking

def cluster_to_cell(cluster, clust_ids):
	"""
	return the cell indices to use in metric calculation for a cluster
	"""
	cell_idx = []
	if cluster == 1:
		cell_idx = np.arange(0,100);
	elif cluster == 2:
		cell_idx = np.arange(100,200);
	elif cluster == 3:
		cell_idx = np.arange(200,300);
	elif cluster == 4:
		cell_idx = np.arange(300,400);
	elif cluster == 5:
		cell_idx = np.arange(400,500);
	assert len(cell_idx) == 100;

	return cell_idx

def cluster_pair_to_cell(cluster1, cluster2, clust_ids):
	"""
	return the cell indices to use in metric calculation for a cluster pair
	"""
	pair_idx = np.append(cluster_to_cell(cluster1, clust_ids), cluster_to_cell(cluster2, clust_ids)).astype(int);
	assert pair_idx.shape[0] == 200;
	return pair_idx

def read_result(result, method):
	"""
	Reads in the result table and sorts genes
	according to importance metric
	Returns the sorted gene indices
	"""
	if method in ['grs', 'deseq2', 'jsd']:
		res = np.genfromtxt(result, delimiter=',', skip_header=1, dtype='unicode')
	if method in ['gce', 'ace', 'sal', 'shap']:
		res = np.genfromtxt(result, delimiter=',', skip_header=0, dtype='unicode')	
	return res

def main(args):
	num_clusters = 5;
	norm_data, clust_labels = normalize_raw_data(args.raw_data)
	idx_lookup_table = make_idx_lookup(args.raw_data)
	print("Dimensions of full data: ", norm_data.shape)

	# what is the mean jaccard distance if using the 20 causal genes?
	baseline = []
	if args.type == 'one_rest':
		for clust in range(1,num_clusters+1):	
			cell_indices = cluster_to_cell(clust, clust_labels);
			baseline.append(calc_jaccard_distance(norm_data, norm_data[:,:20], cell_indices))
		print("Baseline: ", np.mean(baseline))
	else:
		for clust1 in range(1,num_clusters+1):
			for clust2 in range(1,num_clusters+1):
				if clust1 != clust2 and clust2 > clust1:
					cell_indices = cluster_pair_to_cell(clust1, clust2, clust_labels);
					baseline.append(calc_jaccard_distance(norm_data, norm_data.T[:20].T, cell_indices))
		print("Baseline: ", np.mean(baseline))

	jd = {}
	full_res = read_result(args.result, args.method)
	if args.type == 'one_rest':
		# one vs. rest
		for i in range(1,50):#norm_data.shape[1]+1):
			print("number of markers: ", str(i))
			store_jd = []
			for clust in range(1,num_clusters+1):
				gene_rank = obtain_ranking_one_vs_rest(full_res, args.method, clust, idx_lookup_table);
				cell_indices = cluster_to_cell(clust, clust_labels);
				store_jd.append(calc_jaccard_distance(norm_data, norm_data.T[gene_rank][:i].T, cell_indices));
			jd[i] = np.mean(store_jd)
			print("mean jaccard distance: ", str(jd[i]))
		# write out one vs. rest results
		with open(os.path.join(args.output_dir, str(args.method + '_' + args.tag + '_one_vs_rest_jaccard.csv')), 'w') as csv_file:  
			writer = csv.writer(csv_file)
			for key, value in jd.items():
				writer.writerow([key, value])
	elif args.type == 'one_one':
		# one vs. one
		for i in range(1,50): #norm_data.shape[1]+1):
			store_jd = []
			print("number of markers: ", str(i))
			if args.method != 'gce':
				for clust1 in range(1,num_clusters+1):
					for clust2 in range(1,num_clusters+1):
						if clust1 != clust2 and clust2 > clust1:
							gene_rank = obtain_ranking_one_vs_one(full_res, args.method, clust1, clust2, idx_lookup_table);
							cell_indices = cluster_pair_to_cell(clust1, clust2, clust_labels);
							store_jd.append(calc_jaccard_distance(norm_data, norm_data.T[gene_rank][:i].T, cell_indices));
			else:
				for clust in range(2,num_clusters+1):
					gene_rank = obtain_ranking_one_vs_one(full_res, args.method, 1, clust, idx_lookup_table);
					cell_indices = cluster_pair_to_cell(1, clust, clust_labels);
					store_jd.append(calc_jaccard_distance(norm_data, norm_data.T[gene_rank][:i].T, cell_indices));	
			jd[i] = np.mean(store_jd)
			print("mean jaccard distance: ", str(jd[i]))
		# write out one vs. one results
		with open(os.path.join(args.output_dir, str(args.method + '_' + args.tag + '_one_vs_one_jaccard.csv')), 'w') as csv_file:  
			writer = csv.writer(csv_file)
			for key, value in jd.items():
				writer.writerow([key, value])

'''
def main2(args):
	norm_data, clust_labels = normalize_raw_data(args.raw_data)
	idx_lookup_table = make_idx_lookup(args.raw_data)
	print("Dimensions of full data: ", norm_data.shape)

	if args.method == 'baseline':
		print(calc_jaccard_distance(norm_data, norm_data[:,:20]))
	else: 
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
			#print("mean silhouette score: ", str(sil[i]))
			#print("mean jaccard distance: ", str(jd[i]))
			#print("mean FOSCTTM: ", str(foscttm[i]))

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
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--raw_data', type=str, help='raw data');
    parser.add_argument('--result', type=str, help='result');
    parser.add_argument('--method', type=str, help='method');
    parser.add_argument('--output_dir',type=str,help='output directory');
    parser.add_argument('--type',type=str,help='one_rest or one_one');
    parser.add_argument('--tag',type=str,help='tag',default='');

    args = parser.parse_args();
    main(args);	

### python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/clean/one_vs_rest/deseq2_one_vs_rest.csv --method deseq2 --output_dir ../../results/metrics --type one_rest --tag clean
### python3 calc_metrics.py --raw_data ../../data/220var_clean.csv --result ../../results/clean/one_vs_one/deseq2_one_vs_one.csv --method deseq2 --output_dir ../../results/metrics --type one_one --tag clean




