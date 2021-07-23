### run_grscore.R uses destiny's diffusion map embedding to calculate a global gene relevance score for each gene

#install.packages("destiny", repos = "http://cran.us.r-project.org")
#install.packages("tidyverse", repos = "http://cran.us.r-project.org")
#install.packages("scran", repos = "http://cran.us.r-project.org")

setwd("~/Desktop/PBMC/src")

suppressMessages(library(destiny))
suppressMessages(library(tidyverse))
suppressMessages(library(scran))

format_data <- function() {
  # raw data
  raw_data = read.table('../data/3K_PBMC_counts.txt', header = FALSE, sep = ',')+1
  gene_ids = read.table('../data/3K_PBMC_gene_ids.txt', header = FALSE)$V1
  cell_ids = read.table('../data/3K_PBMC_cell_ids.txt', header = FALSE)$V1
  colnames(raw_data) = gene_ids
  rownames(raw_data) = cell_ids
  
  # metadata
  #clust_ids = read.table('../data/3K_PBMC_clust_ids_leiden.txt', header = FALSE)$V1
  clust_ids = read.table('../data/3K_PBMC_clust_ids_kmeans.txt', header = FALSE)$V1
  metadata = data.frame("cellid" = cell_ids, "cluster" = as.factor(clust_ids), stringsAsFactors = FALSE)
  rownames(metadata) = rownames(raw_data)
  
  # convert data into single cell experiment object and log normalize with pseudocount
  sce <- SingleCellExperiment(list(counts=log10(raw_data+1)), metadata=metadata)
  return(sce)
}

extractPartials <- function(sce) {
  # calculate diffusion map embedding using cosine distance
  dm = DiffusionMap(counts(sce), distance = "cosine", n_pcs = 50)
  # calculate global gene relevance for embedding
  gr = gene_relevance(dm)
  # extract partials
  partials_norm = as.data.frame(t(gr@partials_norm))
  return(partials_norm)
}

local_relevance <- function(partials_of_int, top_n, clust_id) {
  # store all genes in top k
  genes_in_top_n = vector()
  for(i in 1:ncol(partials_of_int)) { # i represents a specific cell (ex. 1_1, 2_1, 3_1 ...)
    col = rownames(dplyr::select(partials_of_int, i) %>% top_n(n = top_n))
    genes_in_top_n = append(genes_in_top_n, col)
  }
  # find frequency of genes showing up in top k, divide by total cells in cluster
  local_relevance = data.frame("gene_short_name" = genes_in_top_n) %>% group_by(gene_short_name) %>%
    mutate(freq = n()) %>% distinct() %>% mutate(local_relevance = freq/ncol(partials_of_int)) %>% 
    dplyr::select(-freq) %>% mutate(cluster = clust_id)
  
  # set local relevance of other genes to 0
  other_genes = data.frame("gene_short_name" = setdiff(colnames(data_obj), local_relevance$gene_short_name), 
                           "local_relevance" = 0, "cluster" = clust_id, stringsAsFactors = F)
  # combine the 0's to the local relevance table
  res = rbind(local_relevance, other_genes)
  return(res)
}

# this function performs a one vs. rest comparison (i.e. difference between local relevance of source and complement clusters)
gene_relevance_one_vs_rest <- function(partials, data_obj, top_n) {
  clusters = unique(data_obj@metadata$cluster)
  res = data.frame("cluster" = NULL, "gene_short_name" = NULL, "local_relevance_diff" = NULL, stringsAsFactors = F)
  for(clust_id in clusters) {
    print(paste("Calculating local gene relevance difference for cluster ", clust_id))
    # extract the cells and compute local relevance for the source cluster
    source_idx = which(data_obj@metadata$cluster == clust_id)
    source_partials = partials %>% dplyr::select(all_of(source_idx))
    source_relevance = local_relevance(source_partials, top_n, clust_id)
    
    # extract the cells and compute local relevance for the complement clusters
    target_idx = which(data_obj@metadata$cluster != clust_id)
    target_partials = partials %>% dplyr::select(all_of(target_idx))
    target_relevance = local_relevance(target_partials, top_n, clust_id)
  
    # combine dataframes and take their absolute difference
    combined = left_join(source_relevance, (target_relevance %>% select(-cluster)), by = "gene_short_name") %>% 
      mutate(local_relevance_diff = abs(local_relevance.x - local_relevance.y)) %>% 
      select(cluster, gene_short_name, local_relevance_diff)
    res = rbind(res, combined)
  }
  return(res)
}

# ========================================================================
data_obj = format_data()
partials = extractPartials(data_obj)
write.csv(partials, '../supp_data/grs_partials_kmeans.csv', row.names = TRUE)
res_one_vs_rest = gene_relevance_one_vs_rest(partials, data_obj, 10)
write.csv(res_one_vs_rest, '../results/grs_kmeans.csv', row.names=FALSE)

print("results written to 'results' folder!")