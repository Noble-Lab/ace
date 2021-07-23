### run_grscore.R uses destiny's diffusion map embedding to calculate a global gene relevance score for each gene

#install.packages("destiny", repos = "http://cran.us.r-project.org")
#install.packages("tidyverse", repos = "http://cran.us.r-project.org")
#install.packages("scran", repos = "http://cran.us.r-project.org")

#setwd('~/Desktop/simulation/src/analysis')

suppressMessages(library(destiny))
suppressMessages(library(tidyverse))
suppressMessages(library(scran))

format_data <- function(raw_data, is_batch) {
  if(is_batch == 1) {
    metadata = data.frame("unique_id" = rownames(raw_data)) %>% separate(col = "unique_id", into = c("cellid", "cluster", "batch"), sep = "_")
  } else {
    metadata = data.frame("unique_id" = rownames(raw_data)) %>% separate(col = "unique_id", into = c("cellid", "cluster"), sep = "_")
  }
  rownames(metadata) = rownames(raw_data)
  # convert data into single cell experiment object and log normalize with pseudocount
  sce <- SingleCellExperiment(list(counts=log10(raw_data+1)), metadata=metadata)
  return(sce)
}

extractPartials <- function(sce) {
  # calculate diffusion map embedding using cosine distance
  dm = DiffusionMap(counts(sce), distance = "cosine", n_pcs = 30)
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

# this function performs a one vs. one comparison (i.e. difference between one vs. rest results for two clusters)
gene_relevance_one_vs_one <- function(one_vs_rest_res) {
  comps = combn(x=unique(data_obj@metadata$cluster), m=2)
  res = data.frame("cluster1" = NULL, "cluster2" = NULL, "gene_short_name" = NULL, "diff" = NULL, stringsAsFactors = F)
  for(i in 1:dim(comps)[2]) {
    source = filter(one_vs_rest_res, cluster == comps[,i][1])
    target = filter(one_vs_rest_res, cluster == comps[,i][2])
    combined = left_join(source, target, by = "gene_short_name") %>% 
      mutate(cluster1 = comps[,i][1], cluster2 = comps[,i][2],
             diff = abs(local_relevance_diff.x - local_relevance_diff.y)) %>%
      select(cluster1, cluster2, gene_short_name, diff)
    res = rbind(res, combined)
  }
  return(res)
}

# this function performs local gene relevance for a cluster of interest without making comparisons
local_gene_relevance <- function(partials, data_obj, top_n) {
  clusters = unique(data_obj@metadata$cluster)
  res = data.frame("gene_short_name" = NULL, "local_relevance" = NULL, "cluster" = NULL, stringsAsFactors = F)
  for(clust_id in clusters) {
    print(paste("Calculating local gene relevance for cluster ", clust_id))
    # extract the cells in the specific cluster
    idx = which(data_obj@metadata$cluster == clust_id)
    partials_sub = partials %>% dplyr::select(all_of(idx))
    clust_res = local_relevance(partials_sub, top_n, clust_id)
    res = rbind(res, clust_res)
  }
  res = res %>% group_by(gene_short_name) %>% dplyr::select(-clust_id) %>% mutate(local_relevance = max(local_relevance)) %>%
    ungroup() %>% distinct()
  return(res)
}

# ========================================================================
# clean dataset
raw_data = read.csv('../../data/220var_clean.csv', header = TRUE, row.names = 1)
data_obj = format_data(raw_data, 0)
partials = extractPartials(data_obj)
res_one_vs_rest = gene_relevance_one_vs_rest(partials, data_obj, 10)
res_one_vs_one = gene_relevance_one_vs_one(res_one_vs_rest)
#res = local_gene_relevance(partials, data_obj, 10)

#agg_one_vs_rest = res_one_vs_rest %>% group_by(gene_short_name) %>% mutate(local_relevance_diff = max(local_relevance_diff)) %>% 
#  ungroup() %>% select(gene_short_name, local_relevance_diff) %>% distinct()
#agg_one_vs_one = res_one_vs_one %>% group_by(gene_short_name) %>% mutate(diff = max(diff)) %>% 
#  ungroup() %>% select(gene_short_name, diff) %>% distinct()

# save result
write.csv(res_one_vs_rest, '../../results/clean/one_vs_rest/grs_one_vs_rest.csv', row.names = FALSE)
write.csv(res_one_vs_one, '../../results/clean/one_vs_one/grs_one_vs_one.csv', row.names = FALSE)

# hard dataset
raw_data = read.csv('../../data/220var_hard.csv', header = TRUE, row.names = 1)
data_obj = format_data(raw_data, 1)
partials = extractPartials(data_obj)
res_one_vs_rest = gene_relevance_one_vs_rest(partials, data_obj, 10)
res_one_vs_one = gene_relevance_one_vs_one(res_one_vs_rest)
#res = local_gene_relevance(partials, data_obj, 10)

#agg_one_vs_rest = res_one_vs_rest %>% group_by(gene_short_name) %>% mutate(local_relevance_diff = max(local_relevance_diff)) %>% 
#  ungroup() %>% select(gene_short_name, local_relevance_diff) %>% distinct()
#agg_one_vs_one = res_one_vs_one %>% group_by(gene_short_name) %>% mutate(diff = max(diff)) %>% 
#  ungroup() %>% select(gene_short_name, diff) %>% distinct()

# save result
write.csv(res_one_vs_rest, '../../results/hard/one_vs_rest/grs_one_vs_rest.csv', row.names = FALSE)
write.csv(res_one_vs_one, '../../results/hard/one_vs_one/grs_one_vs_one.csv', row.names = FALSE)

print("results written to 'results' folder!")



