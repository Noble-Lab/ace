import os, sys, argparse
import numpy as np
import pandas as pd
from picturedRocks import Rocks


def get_data(args):
    cell_ids = np.genfromtxt(args.input_data, delimiter=',', skip_header=1, usecols=(0), dtype=None, encoding='ascii');
    cell_ids = np.asarray(cell_ids, dtype=str)

    gene_ids = np.genfromtxt(args.input_data, delimiter=',', max_rows=1, dtype=None);
    gene_ids = np.asarray(gene_ids[1:], dtype=str)

    cell_profiles = np.genfromtxt(args.input_data, delimiter=',', skip_header=1, encoding='ascii');
    cell_profiles = np.asarray(cell_profiles[:, 1:], dtype=float);

    clust_ids = np.asarray([int(id.replace('"', '').split('_')[1]) for id in cell_ids], dtype=int);
    clust_ids = clust_ids - 1 # clusters must be integers starting from 0

    return cell_profiles, cell_ids, gene_ids, clust_ids;

def get_markers(profiles, clust_info, gene_info):
	data = Rocks(profiles, clust_info)
	lamb = 8 # sparsity parameter
	#markers = data.CSrankMarkers(lamb=lamb, writeOut=False, keepZeros=False, onlyNonZero=False)
	markers = data.findCSmarkersForClust(currLamb = lamb, clust = 1)
	print(markers)
	data.genes = gene_info # add gene name info to the Rocks object
	marker_genes = data.markers_to_genes(markers)
	print(marker_genes)

def main(args):
	cell_profiles, cell_ids, gene_ids, clust_ids = get_data(args);
	get_markers(cell_profiles, clust_ids, gene_ids)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Optional app description');
	parser.add_argument('--input_data', type=str, help='input_data');
	args = parser.parse_args();
	main(args);

### to run: python analysis/run_rankcorr.py --input_data ../data/simulate_20causal100noise.csv