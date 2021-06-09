### author: Timothy C. Yu
### date: 11-01-2020
### simulate_synthetic_data.R generates a synthetic dataset containing 300 cells and 120 genes. The dataset is written into the data folder.
### to run (assuming working directory and script is in the simulator folder) 
### Rscript simulate_synthetic_data.R

#install.packages("SymSim", repos = "http://cran.us.r-project.org")
#install.packages("tidyverse", repos = "http://cran.us.r-project.org")
suppressMessages(library(SymSim))
suppressMessages(library(tidyverse))

# load in the tree data, which specifies the degree of separation between groups
small_tree = read.tree(file = "tree_data/small_tree.txt")

# simulate true counts
print("Simulating true counts for: ")
print("300 cells and 400 genes.")
print("3 clusters, 100 cells each.")
true_counts_res <- SimulateTrueCounts(ncells_total=300, min_popsize=100, i_minpop=3, ngenes=400, nevf = 30, n_de_evf = 20, 
                                      evf_type="discrete", vary="s", Sigma=0.4, phyla=small_tree, randseed=55)

findDEGenesInData <- function(n_cluster) {
  DE_summary_table = data.frame("GroupA" = NULL, "GroupB" = NULL, "num_DE" = NULL, stringsAsFactors = FALSE)
  DE_genes_table = data.frame("GroupA" = NULL, "GroupB" = NULL, "gene_short_name" = NULL, "LFC_theoretical" = NULL, "pval_theoretical" = NULL, "DE" = NULL, stringsAsFactors = FALSE)
  for(groupA in 1:n_cluster) {
    for(groupB in 1:n_cluster) {
      if(groupA != groupB) {
        DEinfo <- getDEgenes(true_counts_res, popA = groupA, popB = groupB)
        DEinfo_table = data.frame("nDiffEVF" = DEinfo$nDiffEVF, "logFC_theoretical" = DEinfo$logFC_theoretical, "wil.p_true_counts" = DEinfo$wil.p_true_counts) %>%
          tibble::rownames_to_column(var = "gene_short_name") %>% mutate(DE = ifelse(nDiffEVF > 0 & abs(logFC_theoretical) > 0.8, "yes", "no"))
        DEinfo_table$gene_short_name = paste('gene', DEinfo_table$gene_short_name, sep = "")
        
        DE_genes = dplyr::filter(DEinfo_table, DE == "yes")
        num_DE = nrow(DE_genes)
        temp = data.frame("GroupA" = groupA, "GroupB" = groupB, "numDE" = num_DE, stringsAsFactors = FALSE)
        DE_summary_table = rbind(DE_summary_table, temp)
        
        temp2 = data.frame("GroupA" = groupA, "GroupB" = groupB, "gene_short_name" = DEinfo_table$gene_short_name, "LFC_theoretical" = DEinfo_table$logFC_theoretical, "pval theoretical" = DEinfo_table$wil.p_true_counts,
                           "DE" = DEinfo_table$DE, stringsAsFactors = FALSE)
        DE_genes_table = rbind(DE_genes_table, temp2)
      } 
      else {
        temp = data.frame("GroupA" = groupA, "GroupB" = groupB, "numDE" = NA, stringsAsFactors = FALSE)
        DE_summary_table = rbind(DE_summary_table, temp)
      }
    }
  }
  return(DE_genes_table)
}

DE_genes_table = findDEGenesInData(n_cluster=3)

# select 10 "causal" genes that are most DE between all 3 groups
print("Selecting 10 genes that are DE between all 3 clusters...")
causal_genes = (DE_genes_table %>% filter(DE == 'yes') %>% group_by(gene_short_name) %>% 
                  mutate(n_DE = n(),
                         mean_absLFC = mean(abs(LFC_theoretical))) %>% 
                  ungroup() %>% filter(n_DE == 6) %>% 
                  select(gene_short_name, mean_absLFC) %>% distinct() %>%
                  slice_max(order_by=mean_absLFC, n=10))$gene_short_name
causal_genes_idx = as.numeric(gsub('gene', '', causal_genes))
causal_counts = t(true_counts_res$counts) %>% as.data.frame() %>% select(all_of(causal_genes_idx))
colnames(causal_counts) = gsub("V", "causal", colnames(causal_counts))

# randomly select 100 genes that are not DE in any group comparison
print("Selecting 100 non-DE genes at random...")
set.seed(66)
useless_genes = sample((DE_genes_table %>% filter(DE == 'no') %>% group_by(gene_short_name) %>% 
                          mutate(n_DE = n()) %>% ungroup() %>% filter(n_DE == 6) %>%
                          select(gene_short_name) %>% distinct())$gene_short_name, 100)
useless_genes_idx = as.numeric(gsub('gene', '', useless_genes))
useless_counts = t(true_counts_res$counts) %>% as.data.frame() %>% select(all_of(useless_genes_idx))
colnames(useless_counts) = gsub("V", "useless", colnames(useless_counts))

# add 10 variables that are linear combinations of the causal genes
# we will add Gaussian noise on with a standard deviation of 0.5
# all variables will be multiplied by a random weight between 0.2 and 1
# ==================
# dep1 ~ x1 + N(n = 300, mean = 0, sd = 0.5)
# dep2 ~ x5 + N(300, 0, 0.5)
# dep3 ~ x2 + x3 + N(300, 0, 0.5)
# dep4 ~ x6 + x7 + N(300, 0, 0.5)
# dep5 ~ x9 + x10 + N(300, 0, 0.5)
# dep6 ~ x3 + x6 + x9 + N(300, 0, 0.5)
# dep7 ~ x2 + x4 + x6 + N(300, 0, 0.5)
# dep8 ~ x6 + x7 + x8 + N(300, 0, 0.5)
# dep9 ~ x1 + x3 + x5 + x7 + x9 + N(300, 0, 0.5)
# dep10 ~ x2 + x4 + x6 + x8 + x10 + N(300, 0, 0.5)
# ==================
generateDependentVariables <- function(causal_data, n) {
  dep1 = floor(runif(1,0.2,1)*causal_data[,1] + rnorm(n,0,0.5))
  dep2 = floor(runif(1,0.2,1)*causal_data[,5] + rnorm(n,0,0.5))
  dep3 = floor(runif(1,0.2,1)*causal_data[,2] + runif(1,0.2,1)*causal_data[,3] + rnorm(n,0,0.5))
  dep4 = floor(runif(1,0.2,1)*causal_data[,6] + runif(1,0.2,1)*causal_data[,7] + rnorm(n,0,0.5))
  dep5 = floor(runif(1,0.2,1)*causal_data[,9] + runif(1,0.2,1)*causal_data[,10] + rnorm(n,0,0.5))
  dep6 = floor(runif(1,0.2,1)*causal_data[,3] + runif(1,0.2,1)*causal_data[,6] + runif(1,0.2,1)*causal_data[,9] + rnorm(n,0,0.5))
  dep7 = floor(runif(1,0.2,1)*causal_data[,2] + runif(1,0.2,1)*causal_data[,4] + runif(1,0.2,1)*causal_data[,6] + rnorm(n,0,0.5))
  dep8 = floor(runif(1,0.2,1)*causal_data[,6] + runif(1,0.2,1)*causal_data[,7] + runif(1,0.2,1)*causal_data[,8] + rnorm(n,0,0.5))
  dep9 = floor(runif(1,0.2,1)*causal_data[,1] + runif(1,0.2,1)*causal_data[,3] + runif(1,0.2,1)*causal_data[,5] +
                 runif(1,0.2,1)*causal_data[,7] + runif(1,0.2,1)*causal_data[,9] + rnorm(n,0,0.5))
  dep10 = floor(runif(1,0.2,1)*causal_data[,2] + runif(1,0.2,1)*causal_data[,4] + runif(1,0.2,1)*causal_data[,6] +
                  runif(1,0.2,1)*causal_data[,8] + runif(1,0.2,1)*causal_data[,10] + rnorm(n,0,0.5))
  result = cbind(dep1, dep2, dep3, dep4, dep5, dep6, dep7, dep8, dep9, dep10)
  return(result)
}

print("Simulating 10 variables that are linear combinations of the DE genes...")
set.seed(5969)
dep_counts = generateDependentVariables(as.data.frame(causal_counts), nrow(causal_counts))

# join to obtain the final dataset
joint_data = cbind(causal_counts, useless_counts, dep_counts)

# save the dataset into data folder
print("Writing dataset to data directory...")
cell_ids = vector()
for(i in 1:300) { cell_ids = append(cell_ids, paste(i, "_", true_counts_res$cell_meta$pop[i], sep = ""))}
rownames(joint_data) = cell_ids
write.csv(joint_data, '../../data/simulated_data_with_dependencies_120var.csv', row.names = TRUE)
print("Done.")
