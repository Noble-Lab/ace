### author: Timothy C. Yu
### date: 11-01-2020
### run_deseq2.R uses DESeq2 to identify DE genes between clusters
### to run (assuming working directory and script is in the analysis folder):
### Rscript run_deseq2.R

#install.packages("DESeq2", repos = "http://cran.us.r-project.org")
suppressMessages(library(DESeq2))
suppressMessages(library(tidyverse))

#setwd('~/Documents/GitHub/2020_ylu465_group_explain/src/analysis')

# format data
get_metadata <- function(raw_data) {
  metadata = data.frame("unique_id" = rownames(raw_data)) %>% separate(col = "unique_id", into = c("cellid", "cluster"), sep = "_")
  rownames(metadata) = rownames(raw_data)
  metadata$cluster = as.factor(metadata$cluster)
  return(metadata)
}

# run DESeq2
get_de_genes <- function(raw_data, meta_data) {
  dds <- DESeqDataSetFromMatrix(t(as.matrix(raw_data)), colData=meta_data, design= ~cluster)
  dds <- DESeq(dds, minReplicatesForReplace=Inf, fitType="mean", betaPrior=TRUE) # argument turns off outlier filtering/replacement
  
  comps = resultsNames(dds)[-1]
  table = data.frame("Group" = NULL, "gene_short_name" = NULL, "pvalue" = NULL, "padj" = NULL, "DE" = NULL, stringsAsFactors = FALSE)
  for(i in 1:length(comps)) {
    res = results(dds, contrast = list(comps[i], comps[-i]), listValues=c(1,-1/length(comps)), cooksCutoff = FALSE)
    data_entry = as.data.frame(res) %>% tibble::rownames_to_column(var = "gene_short_name") %>% 
      mutate(Group = comps[i], DE = ifelse(padj < 0.05, "yes", "no")) %>%
      dplyr::select(Group, gene_short_name, pvalue, padj, DE)
    table = rbind(table, data_entry)
  }
  return(table)
}

# ========================================================================
#raw = read.csv('../../data/simulate_20causal100noise.csv', header = TRUE, row.names = 1)
raw = read.csv('../../data/220var_clean.csv', header = TRUE, row.names = 1)
meta = get_metadata(raw)
res = get_de_genes(raw, meta) %>% group_by(gene_short_name) %>% mutate(padj = min(padj)) %>% ungroup() %>%
  select(gene_short_name, padj) %>% distinct()

# save result
write.csv(res, '../../results/method_output_220var/deseq2_result.csv', row.names = FALSE)
print("results written to 'method_output_220var' folder!")
