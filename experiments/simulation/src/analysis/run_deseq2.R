### run_deseq2.R uses DESeq2 to identify DE genes between clusters in either a one vs. one or one vs. rest fashioin

#install.packages("DESeq2", repos = "http://cran.us.r-project.org")
suppressMessages(library(DESeq2))
suppressMessages(library(tidyverse))

#setwd('~/Desktop/simulation/src/analysis')

# format data
get_metadata <- function(raw_data, is_batch) {
  if(is_batch == 1) {
    metadata = data.frame("unique_id" = rownames(raw_data)) %>% separate(col = "unique_id", into = c("cellid", "cluster", "batch"), sep = "_")
  } else {
    metadata = data.frame("unique_id" = rownames(raw_data)) %>% separate(col = "unique_id", into = c("cellid", "cluster"), sep = "_")
  }
  rownames(metadata) = rownames(raw_data)
  metadata$cluster = as.factor(metadata$cluster)
  return(metadata)
}

# run DESeq2 with one vs. rest
DESeq2_one_vs_rest <- function(raw_data, meta_data) {
  dds <- DESeqDataSetFromMatrix(t(as.matrix(raw_data)), colData=meta_data, design = ~cluster)
  dds <- DESeq(dds, minReplicatesForReplace=Inf, fitType = "mean", betaPrior=TRUE) # argument turns off outlier filtering/replacement
  
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
  #table = table %>% group_by(gene_short_name) %>% mutate(padj = min(padj)) %>% ungroup() %>%
  #  select(gene_short_name, padj) %>% distinct() %>%
  #  replace(is.na(.), 1)
  return(table)
}

# run DESeq2 with one vs. one
DESeq2_one_vs_one <- function(raw_data, meta_data) {
  dds <- DESeqDataSetFromMatrix(t(as.matrix(raw_data)), colData=meta_data, design= ~cluster)
  dds <- DESeq(dds, minReplicatesForReplace=Inf, fitType = "mean", betaPrior=TRUE) # argument turns off outlier filtering/replacement
  
  table = data.frame("cluster1" = NULL, "cluster2" = NULL, "gene_short_name" = NULL, "padj" = NULL, stringsAsFactors = FALSE)
  comps = combn(x=unique(meta_data$cluster), m=2)
  for(i in 1:dim(comps)[2]) {
    res = results(dds, contrast=c("cluster", comps[,i][1], comps[,i][2]))
    data_entry = as.data.frame(res) %>% tibble::rownames_to_column(var="gene_short_name") %>% 
      mutate(cluster1 = comps[,i][1], cluster2 = comps[,i][2]) %>%
      select(cluster1, cluster2, gene_short_name, padj)
    table = rbind(table, data_entry)
  }
  table = table %>% replace(is.na(.), 1)
  #table = table %>% group_by(gene_short_name) %>% mutate(padj = min(padj)) %>% ungroup() %>%
  #  select(gene_short_name, padj) %>% distinct() %>%
  #  replace(is.na(.), 1)
  return(table)
}  

# ========================================================================
# clean dataset
raw = read.csv('../../data/220var_clean.csv', header = TRUE, row.names = 1)
meta = get_metadata(raw, 0)
one_vs_rest_res = DESeq2_one_vs_rest(raw, meta)
one_vs_one_res = DESeq2_one_vs_one(raw, meta)

# save result
write.csv(one_vs_rest_res, '../../results/clean/one_vs_rest/deseq2_one_vs_rest.csv', row.names = FALSE)
write.csv(one_vs_one_res, '../../results/clean/one_vs_one/deseq2_one_vs_one.csv', row.names = FALSE)

# hard dataset
raw = read.csv('../../data/220var_hard.csv', header = TRUE, row.names = 1)
meta = get_metadata(raw, 1)
one_vs_rest_res = DESeq2_one_vs_rest(raw+1, meta)
one_vs_one_res = DESeq2_one_vs_one(raw+1, meta)

# save result
write.csv(one_vs_rest_res, '../../results/hard/one_vs_rest/deseq2_one_vs_rest.csv', row.names = FALSE)
write.csv(one_vs_one_res, '../../results/hard/one_vs_one/deseq2_one_vs_one.csv', row.names = FALSE)

print("results written to 'results' folder!")