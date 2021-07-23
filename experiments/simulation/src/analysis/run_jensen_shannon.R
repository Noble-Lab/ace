#install.packages("monocle3", repos = "http://cran.us.r-project.org")
#install.packages("tidyverse", repos = "http://cran.us.r-project.org")
suppressMessages(library(monocle3))
suppressMessages(library(tidyverse))

#setwd('~/Desktop/simulation/src/analysis')

# format data
preprocess_data <- function(raw_data, is_batch) {
  if(is_batch == 1) {
    metadata = data.frame("unique_id" = rownames(raw_data)) %>% separate(col = "unique_id", into = c("cellid", "cluster", "batch"), sep = "_")
  } else {
    metadata = data.frame("unique_id" = rownames(raw_data)) %>% separate(col = "unique_id", into = c("cellid", "cluster"), sep = "_")
  }
  rownames(metadata) = rownames(raw_data)
  gene_metadata = data.frame("gene_short_name" = colnames(raw_data), stringsAsFactors = FALSE, row.names = colnames(raw_data))
  
  # load up data into Monocle 3's main class, cell_data_set
  cds <- new_cell_data_set(t(as.matrix(raw_data)),
                           cell_metadata = metadata,
                           gene_metadata = gene_metadata)
  # normalize and pre-process the data. generate PCA components.
  cds <- preprocess_cds(cds, num_dim = 30)
  # reduce the dimensions using UMAP
  cds <- reduce_dimension(cds, preprocess_method = "PCA")
  return(list("obj" = cds, "meta" = metadata))
}

# this function implements jensen-shannon between a source cluster and all complement clusters (modeled as a single cluster). 
jensen_shannon_one_vs_rest <- function(cds) {
  clusters = unique(cds$meta$cluster)
  res = data.frame("gene_short_name" = NULL, "marker_test_q_value" = NULL, "cluster" = NULL, stringsAsFactors = F)
  for(clust_id in clusters) {
    # make cluster of interest as 1, all other clusters as 0
    mod_cluster_meta = (cds$meta %>% mutate(mod_cluster = ifelse(cluster == clust_id, 1, 0)))$mod_cluster
    cds$obj@clusters$UMAP$clusters = as.factor(mod_cluster_meta)
    names(cds$obj@clusters$UMAP$clusters) = rownames(cds$meta)
    
    # find markers specific to clust_id relative to the complement set
    marker_test_res <- top_markers(cds$obj, group_cells_by="cluster", cores=1, genes_to_test_per_group=100, speedglm.maxiter = 100) %>%
      filter(cell_group == 1) %>% mutate(cluster = clust_id) %>% select(gene_short_name, marker_test_q_value, cluster)
    
    res = rbind(marker_test_res, res)
  }
  #agg_res = res %>% group_by(gene_short_name) %>% mutate(q_value = min(marker_test_q_value)) %>% ungroup() %>% 
  #  select(gene_short_name, q_value) %>% distinct()
  all_res = data.frame("gene_short_name" = NULL, "marker_test_q_value" = NULL, "cluster" = NULL, stringsAsFactors = F)
  for(clust_id in clusters) {
    res_genes = filter(res, cluster == clust_id)
    no_res_genes = data.frame("gene_short_name" = setdiff(colnames(raw_data), res_genes$gene_short_name), 
                              "marker_test_q_value" = 1, "cluster" = clust_id)
    all_res = rbind(all_res, res_genes, no_res_genes)
  }
  #no_res_genes = data.frame("gene_short_name" = setdiff(colnames(raw_data), agg_res$gene_short_name), "q_value" = 1)
  #agg_res = rbind(agg_res, no_res_genes)
  return(all_res %>% select(cluster, gene_short_name, marker_test_q_value))
}

# this function implements jensen-shannon between all pairwise cluster comparisons
jensen_shannon_one_vs_one <- function(cds) {
  comps = combn(x=unique(cds$meta$cluster), m=2)
  res = data.frame("gene_short_name" = NULL, "marker_test_q_value" = NULL, "cluster1" = NULL, "cluster2" = NULL, stringsAsFactors = F)
  for(i in 1:dim(comps)[2]) {
    # modify the cluster metadata to only include the two clusters of interest
    mod_cluster_meta = (cds$meta %>% filter(cluster %in% c(comps[,i][1],comps[,i][2])))$cluster
    cds$obj@clusters$UMAP$clusters = as.factor(mod_cluster_meta)
    cluster_idx = which(cds$meta$cluster %in% c(comps[,i][1],comps[,i][2]))
    names(cds$obj@clusters$UMAP$clusters) = rownames(cds$meta)[cluster_idx]
    
    # modify the actual cds object to only include measurements from the cells in the two clusters of interest
    cds_subset = cds$obj[,cluster_idx]
    
    # find markers specific to each of the two clusters
    marker_test_res <- top_markers(cds_subset, group_cells_by="cluster", cores=1, genes_to_test_per_group=100, speedglm.maxiter = 100) %>%
      mutate("cluster1" = comps[,i][1], "cluster2" = comps[,i][2]) %>%
      select(gene_short_name, marker_test_q_value, cluster1, cluster2)
    res = rbind(marker_test_res, res)
  }
  
  all_res = data.frame("gene_short_name" = NULL, "marker_test_q_value" = NULL, "cluster1" = NULL, "cluster2" = NULL, stringsAsFactors = F)
  for(i in 1:dim(comps)[2]) {
      res_genes = filter(res, cluster1 == comps[,i][1], cluster2 == comps[,i][2]) %>% group_by(gene_short_name) %>%
        mutate(marker_test_q_value = min(marker_test_q_value)) %>% distinct()
      no_res_genes = data.frame("gene_short_name" = setdiff(colnames(raw_data), res_genes$gene_short_name), 
                                "marker_test_q_value" = 1, "cluster1" = comps[,i][1], "cluster2" = comps[,i][2])
      all_res = rbind(all_res, res_genes, no_res_genes)
  }
  #agg_res = res %>% group_by(gene_short_name) %>% mutate(q_value = min(marker_test_q_value)) %>% ungroup() %>% 
  #  select(gene_short_name, q_value) %>% distinct()
  #no_res_genes = data.frame("gene_short_name" = setdiff(colnames(raw_data), agg_res$gene_short_name), "q_value" = 1)
  #agg_res = rbind(agg_res, no_res_genes)
  return(all_res %>% select(cluster1, cluster2, gene_short_name, marker_test_q_value))
}

# ========================================================================
raw_data = read.csv('../../data/220var_clean.csv', header = TRUE, row.names = 1)
data_obj = preprocess_data(raw_data, 0)
res_one_vs_rest = jensen_shannon_one_vs_rest(data_obj)
res_one_vs_one = jensen_shannon_one_vs_one(data_obj)

# save result
write.csv(res_one_vs_rest, '../../results/clean/one_vs_rest/jsd_one_vs_rest.csv', row.names = FALSE)
write.csv(res_one_vs_one, '../../results/clean/one_vs_one/jsd_one_vs_one.csv', row.names = FALSE)

raw_data = read.csv('../../data/220var_hard.csv', header = TRUE, row.names = 1)
data_obj = preprocess_data(raw_data, 1)
res_one_vs_rest = jensen_shannon_one_vs_rest(data_obj)
res_one_vs_one = jensen_shannon_one_vs_one(data_obj)

# save result
write.csv(res_one_vs_rest, '../../results/hard/one_vs_rest/jsd_one_vs_rest.csv', row.names = FALSE)
write.csv(res_one_vs_one, '../../results/hard/one_vs_one/jsd_one_vs_one.csv', row.names = FALSE)
print("results written to 'results' folder!")
