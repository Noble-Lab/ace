### run_deseq2.R uses DESeq2 to identify DE genes between clusters in either a one vs. one or one vs. rest fashioin

#install.packages("DESeq2", repos = "http://cran.us.r-project.org")
suppressMessages(library(DESeq2))
suppressMessages(library(tidyverse))

setwd("~/Desktop/PBMC/src")

# format data
get_metadata <- function() {
  cell_ids = read.table('../data/3K_PBMC_cell_ids.txt', header = FALSE)$V1
  #clust_ids = read.table('../data/3K_PBMC_clust_ids_leiden.txt', header = FALSE)$V1
  clust_ids = read.table('../data/3K_PBMC_clust_ids_kmeans.txt', header = FALSE)$V1
  metadata = data.frame("cellid" = cell_ids, "cluster" = as.factor(clust_ids), stringsAsFactors = FALSE)
  return(metadata)
}

get_rawdata <- function() {
  raw_data = read.table('../data/3K_PBMC_counts.txt', header = FALSE, sep = ',')
  gene_ids = read.table('../data/3K_PBMC_gene_ids.txt', header = FALSE)$V1
  colnames(raw_data) = gene_ids
  return(raw_data)
}

# run DESeq2 with one vs. rest
DESeq2_one_vs_rest <- function(raw_data, meta_data) {
  dds <- DESeqDataSetFromMatrix(t(as.matrix(raw_data)), colData=meta_data, design = ~cluster)
  dds <- DESeq(dds, minReplicatesForReplace=Inf, betaPrior=TRUE) # argument turns off outlier filtering/replacement
  
  comps = resultsNames(dds)[-1]
  table = data.frame("cluster" = NULL, "gene_short_name" = NULL, "pvalue" = NULL, "padj" = NULL, "DE" = NULL, stringsAsFactors = FALSE)
  for(i in 1:length(comps)) {
    res = results(dds, contrast = list(comps[i], comps[-i]), listValues=c(1,-1/length(comps)))
    data_entry = as.data.frame(res) %>% tibble::rownames_to_column(var = "gene_short_name") %>% 
      mutate(cluster = comps[i], DE = ifelse(padj < 0.05, "yes", "no")) %>%
      dplyr::select(cluster, gene_short_name, pvalue, padj, DE)
    table = rbind(table, data_entry)
  }
  table = table %>% select(cluster, gene_short_name, padj) %>% replace(is.na(.), 1)
  table$cluster = gsub("cluster", "", table$cluster)
  return(table)
}

# ========================================================================
raw = get_rawdata()
meta = get_metadata()
one_vs_rest_res = DESeq2_one_vs_rest(raw+1, meta)

# save result
#write.csv(one_vs_rest_res, '../results/deseq2_leiden.csv', row.names=FALSE)
write.csv(one_vs_rest_res, '../results/deseq2_kmeans.csv', row.names=FALSE)

print("results written to 'results' folder!")