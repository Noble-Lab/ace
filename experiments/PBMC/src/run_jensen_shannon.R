#install.packages("monocle3", repos = "http://cran.us.r-project.org")
#install.packages("tidyverse", repos = "http://cran.us.r-project.org")
suppressMessages(library(monocle3))
suppressMessages(library(tidyverse))

setwd("~/Desktop/PBMC/src")

# format data
preprocess_data <- function() {
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
  gene_metadata = data.frame("gene_short_name" = gene_ids, stringsAsFactors = FALSE, row.names = gene_ids)
  
  # load up data into Monocle 3's main class, cell_data_set
  cds <- new_cell_data_set(t(as.matrix(raw_data)),
                           cell_metadata = metadata,
                           gene_metadata = gene_metadata)
  # normalize and pre-process the data. generate PCA components.
  cds <- preprocess_cds(cds, num_dim = 50)
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
    marker_test_res <- top_markers(cds$obj, group_cells_by="cluster", cores=1, genes_to_test_per_group=300, speedglm.maxiter = 100) %>%
      filter(cell_group == 1) %>% mutate(cluster = clust_id) %>% select(gene_short_name, marker_test_q_value, cluster)
    res = rbind(marker_test_res, res)
  }
  all_res = data.frame("gene_short_name" = NULL, "marker_test_q_value" = NULL, "cluster" = NULL, stringsAsFactors = F)
  for(clust_id in clusters) {
    res_genes = filter(res, cluster == clust_id)
    no_res_genes = data.frame("gene_short_name" = setdiff(rownames(cds$obj), res_genes$gene_short_name), 
                              "marker_test_q_value" = 1, "cluster" = clust_id)
    all_res = rbind(all_res, res_genes, no_res_genes)
  }
  return(all_res %>% select(cluster, gene_short_name, marker_test_q_value))
}

# ========================================================================
data_obj = preprocess_data()
res_one_vs_rest = jensen_shannon_one_vs_rest(data_obj)

# save result
#write.csv(res_one_vs_rest, '../results/jsd_leiden.csv', row.names=FALSE)
write.csv(res_one_vs_rest, '../results/jsd_kmeans.csv', row.names=FALSE)

print("results written to 'results' folder!")
