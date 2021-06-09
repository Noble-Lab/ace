### THIS CODE IS NOT COMPLETE. PLEASE DO NOT USE RIGHT NOW ###
### author: Timothy C. Yu
### date: 11-01-2020
### run_grscore.R uses destiny's diffusion map embedding to calculate a global gene relevance score for each gene
### to run (assuming working directory and script is in the analysis folder):
### Rscript run_grscore.R

#install.packages("destiny", repos = "http://cran.us.r-project.org")
#install.packages("tidyverse", repos = "http://cran.us.r-project.org")
#install.packages("scran", repos = "http://cran.us.r-project.org")

suppressMessages(library(destiny))
suppressMessages(library(tidyverse))
suppressMessages(library(scran))

format_data <- function(raw_data) {
  metadata = data.frame("unique_id" = rownames(raw_data)) %>% separate(col = "unique_id", into = c("cellid", "cluster"), sep = "_")
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

calculateLocalGeneRelevance <- function(partials, data_obj, top_n) {
  clusters = unique(data_obj@metadata$cluster)
  res = data.frame("genes" = NULL, "local_relevance" = NULL, "clust_id" = NULL, stringsAsFactors = F)
  for(clust_id in clusters) {
    print(paste("Calculating local gene relevance for cluster ", clust_id))
    # extract the cells in the specific cluster
    idx = which(data_obj@metadata$cluster == clust_id)
    partials_sub = partials %>% dplyr::select(all_of(idx))
    # store all genes in top k
    genes_in_top_n = vector()
    for(i in 1:ncol(partials_sub)) { # i represents a specific cell (ex. 1_1, 2_1, 3_1 ...)
      col = rownames(dplyr::select(partials_sub, i) %>% top_n(n = top_n))
      genes_in_top_n = append(genes_in_top_n, col)
    }
    # find frequency of genes showing up in top k, divide by total cells
    local_relevance = data.frame("genes" = genes_in_top_n) %>% group_by(genes) %>%
      mutate(freq = n()) %>% distinct() %>%
      mutate(local_relevance = freq/ncol(partials_sub)) %>% dplyr::select(-freq) %>% mutate(clust_id = clust_id)
    # set local relevance of other genes to 0
    other_genes = data.frame("genes" = setdiff(colnames(data_obj), local_relevance$genes), "local_relevance" = 0,
                             "clust_id" = clust_id, stringsAsFactors = F)
    clust_res = rbind(local_relevance, other_genes)
    res = rbind(res, clust_res)
  }
  res = res %>% group_by(genes) %>% dplyr::select(-clust_id) %>% mutate(local_relevance = max(local_relevance)) %>%
    ungroup() %>% distinct()
  return(res)
}

# ========================================================================
# raw_data = read.csv('../../data/simulate_20causal100noise.csv', header = TRUE, row.names = 1)
raw_data = read.csv('../../data/220var_clean.csv', header = TRUE, row.names = 1)
data_obj = format_data(raw_data)
partials = extractPartials(data_obj)
res = calculateLocalGeneRelevance(partials, data_obj, 10)

# save result
write.csv(res, '../../results/method_output_220var/grscore_result.csv', row.names = FALSE)
print("results written to 'method_output' folder!")
