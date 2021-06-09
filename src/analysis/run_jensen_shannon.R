#install.packages("monocle3", repos = "http://cran.us.r-project.org")
#install.packages("tidyverse", repos = "http://cran.us.r-project.org")
suppressMessages(library(monocle3))
suppressMessages(library(tidyverse))

# format data
preprocess_data <- function(raw_data) {
  metadata = data.frame("unique_id" = rownames(raw_data)) %>% separate(col = "unique_id", into = c("cellid", "cluster", "batch"), sep = "_")
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
  # using pre-specified clusters
  cds@clusters$UMAP$clusters = as.factor(metadata$cluster)
  names(cds@clusters$UMAP$clusters) = rownames(metadata)
  cds@clusters$UMAP$clusters
  return(cds)
}

# ========================================================================
#raw_data = read.csv('../../data/simulate_20causal100noise.csv', header = TRUE, row.names = 1)
raw_data = read.csv('../../data/220var_clean.csv', header = TRUE, row.names = 1)
data_obj = preprocess_data(raw_data)

# jensen-shannon specificity implementation
marker_test_res <- top_markers(data_obj, group_cells_by="cluster", cores=1, genes_to_test_per_group=50, speedglm.maxiter = 100)
res = marker_test_res %>% group_by(gene_short_name) %>% mutate(q_value = min(marker_test_q_value)) %>% ungroup() %>%
  select(gene_short_name, q_value) %>% distinct()
no_res_genes = data.frame("gene_short_name" = setdiff(colnames(raw_data), res$gene_short_name), "q_value" = 1)
res = rbind(res, no_res_genes)

# save result
write.csv(res, '../../results/method_output_220var/jensen_shannon_result.csv', row.names = FALSE)
print("results written to 'method_output_220var' folder!")