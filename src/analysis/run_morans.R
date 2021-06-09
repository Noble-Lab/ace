### author: Timothy C. Yu
### date: 11-01-2020
### run_morans.R uses Monocle 3's Moran's I test to identify DE genes between clusters
### to run (assuming working directory and script is in the analysis folder):
### Rscript run_morans.R

#install.packages("monocle3", repos = "http://cran.us.r-project.org")
#install.packages("tidyverse", repos = "http://cran.us.r-project.org")
suppressMessages(library(monocle3))
suppressMessages(library(tidyverse))

# format data
preprocess_data <- function(raw_data) {
  metadata = data.frame("unique_id" = rownames(raw_data)) %>% separate(col = "unique_id", into = c("cellid", "cluster"), sep = "_")
  rownames(metadata) = rownames(raw_data)
  gene_metadata = data.frame("gene_short_name" = colnames(raw_data), stringsAsFactors = FALSE, row.names = colnames(raw_data))
  
  # load up data into Monocle 3's main class, cell_data_set
  cds <- new_cell_data_set(t(as.matrix(raw_data)),
                           cell_metadata = metadata,
                           gene_metadata = gene_metadata)
  # normalize and pre-process the data. generate 10 PCA components.
  cds <- preprocess_cds(cds, num_dim = 50)
  # reduce the dimensions using UMAP
  cds <- reduce_dimension(cds, preprocess_method = "PCA")
  # using pre-specified clusters
  colData(cds)$assigned_cell_type = as.character(metadata$cluster)
  return(cds)
}

# run Moran's I test between specific comparisons
get_de_genes <- function(cds) {
  deg_table = data.frame("status" = NULL, "p_value" = NULL, "morans_test_statistic" = NULL, "morans_I" = NULL, "gene_short_name" = NULL, "q_value" = NULL, "comparison" = NULL, stringsAsFactors = FALSE)
  comparisons = combn(unique(colData(cds)$assigned_cell_type), m=2)
  for(i in 1:ncol(comparisons)) {
    # perform the differential expression and obtain Moran's I value and q-value for each comparison
    filtered_cds = cds[,which(colData(data_obj)$assigned_cell_type == comparisons[,i][1] | 
                                 colData(data_obj)$assigned_cell_type == comparisons[,i][2])]
    pr_graph_test_res = graph_test(filtered_cds, neighbor_graph="knn", cores=4) %>% mutate(GroupA = comparisons[,i][1],
                                                                                            GroupB = comparisons[,i][2])
    deg_table = rbind(deg_table, pr_graph_test_res)
  }
  result = deg_table %>% mutate(DE = ifelse(q_value < 0.05, "yes", "no")) %>%
                         dplyr::select(GroupA, GroupB, gene_short_name, morans_I, p_value, q_value, DE)
  return(result)
} 

# run Moran's I on entire UMAP embedding
get_de_genes2 <- function(cds) {
  pr_graph_test_res = graph_test(cds, neighbor_graph="knn", cores=4)
  result = pr_graph_test_res %>% mutate(DE = ifelse(q_value < 0.05, "yes", "no"))
  return(result)
} 
# ========================================================================
#raw_data = read.csv('../../data/simulate_20causal100noise.csv', header = TRUE, row.names = 1)
raw_data = read.csv('../../data/trajectory_clean_counts.csv', header = TRUE, row.names = 1)
data_obj = preprocess_data(raw_data)
res = get_de_genes2(data_obj)

# save result
write.csv(res, '../../results/morans_result.csv', row.names = FALSE)
print("results written to 'results' folder!")