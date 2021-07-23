### simulate_synthetic_data.R generates a synthetic dataset. The dataset is written into the data folder.

#install.packages("SymSim", repos = "http://cran.us.r-project.org")
#install.packages("tidyverse", repos = "http://cran.us.r-project.org")
suppressMessages(library(SymSim))
suppressMessages(library(tidyverse))
suppressMessages(library(uwot))
suppressMessages(require(cowplot))
suppressMessages(theme_set(theme_cowplot()))

setwd('~/Desktop/simulation/src/simulator')

# load in the tree data, which specifies the degree of separation between groups
small_tree = read.tree(file = "tree_data/small_tree.txt")

# simulate true counts
print("Using SymSim to simulate true counts for: ")
print("500 cells and 2000 genes.")
print("5 clusters, 100 cells each.")
true_counts_res <- SimulateTrueCounts(ncells_total=500, min_popsize=100, i_minpop=3, ngenes=2000, nevf = 30, n_de_evf = 20, 
                                      evf_type="discrete", vary="s", Sigma=0.4, phyla=small_tree, randseed=55)

findDEGenesInData <- function(n_cluster, counts) {
  DE_summary_table = data.frame("GroupA" = NULL, "GroupB" = NULL, "num_DE" = NULL, stringsAsFactors = FALSE)
  DE_genes_table = data.frame("GroupA" = NULL, "GroupB" = NULL, "gene_short_name" = NULL, "LFC_theoretical" = NULL, "pval_theoretical" = NULL, "DE" = NULL, stringsAsFactors = FALSE)
  for(groupA in 1:n_cluster) {
    for(groupB in 1:n_cluster) {
      if(groupA != groupB) {
        DEinfo <- getDEgenes(counts, popA = groupA, popB = groupB)
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

DE_genes_table = findDEGenesInData(n_cluster=5, true_counts_res)

# select 20 "causal" genes that are most DE between all 5 groups
print("Selecting 20 causal genes...")
causal_genes = (DE_genes_table %>% filter(DE == 'yes') %>% group_by(gene_short_name) %>% 
                  dplyr::mutate(n_DE = n(),
                         mean_absLFC = mean(abs(LFC_theoretical))) %>% 
                  ungroup() %>%
                  dplyr::select(gene_short_name, mean_absLFC) %>% distinct() %>%
                  slice_max(order_by=mean_absLFC, n=20))$gene_short_name
causal_genes_idx = as.numeric(gsub('gene', '', causal_genes))
causal_counts = t(true_counts_res$counts) %>% as.data.frame() %>% dplyr::select(all_of(causal_genes_idx))
colnames(causal_counts) = gsub("V", "causal", colnames(causal_counts))

# randomly select 100 genes that are not DE in any group comparison
print("Selecting 100 noise genes...")
set.seed(66)
useless_genes = sample((DE_genes_table %>% filter(DE == 'no') %>% group_by(gene_short_name) %>% 
                          dplyr::mutate(n_DE = n()) %>% ungroup() %>% filter(n_DE == 20) %>%
                          dplyr::select(gene_short_name) %>% distinct())$gene_short_name, 100)
useless_genes_idx = as.numeric(gsub('gene', '', useless_genes))
useless_counts = t(true_counts_res$counts) %>% as.data.frame() %>% dplyr::select(all_of(useless_genes_idx))
colnames(useless_counts) = gsub("V", "useless", colnames(useless_counts))

# add 100 variables that are linear combinations of the causal genes
# we will add Gaussian noise on with a standard deviation of 1
# all variables will be multiplied by a random weight between 0.01 and 0.5
# ==================
generateDepVariables <- function(causal_data, num_dep) {
  dep_counts = data.frame(row.names=1:nrow(causal_data))
  col_names = NULL
  for(i in 1:num_dep) {
    # select random number between 1 and 10 (number of causal variables to model from)
    num_var = sample(1:10, 1)
    # randomly select the indices for the causal variables to use
    idx = sample(1:20, num_var)
    # randomly find weights for each of the variables
    weights = runif(num_var, 0.01, 0.8)
    # create dependent variable
    dep = pmax(ceiling(rowSums(t(t(causal_data[,idx])*weights)) + rnorm(nrow(causal_data),0,1)), 0)
    # add to data
    dep_counts = cbind(dep_counts, dep)
    col_names = append(col_names, paste("dep", i, sep=""))
  }
  colnames(dep_counts) = col_names
  return(dep_counts)
}

print("Simulating 100 dependent variables that are correlated to the causal genes...")
set.seed(2020)
dep_counts = generateDepVariables(causal_counts, 100)

# join to obtain the final dataset
joint_data = cbind(causal_counts, useless_counts, dep_counts)

# plotting
print("Plotting data...")

set.seed(1)
umap <- umap(log10(causal_counts+1), init = "spectral", verbose = FALSE)
vis_umap = cbind(as.data.frame(umap), "Group" = as.factor(true_counts_res$cell_meta$pop))

causal = ggplot(vis_umap, aes(x = V1, y = V2)) + geom_point(aes(color = Group), alpha = 0.6, size = 1) +
  scale_color_manual(values = c("#ECA400", "#C81D25", "#00A9A5", "green", "purple"), name = "Group") +
  labs(x = "UMAP1", y = "UMAP2", title = "Causal") + theme(plot.title = element_text(hjust = 0.5)) 

set.seed(2)
umap <- umap(log10(dep_counts+1), init = "spectral", verbose = FALSE)
vis_umap = cbind(as.data.frame(umap), "Group" = as.factor(true_counts_res$cell_meta$pop))

dep = ggplot(vis_umap, aes(x = V1, y = V2)) + geom_point(aes(color = Group), alpha = 0.6, size = 1) +
  scale_color_manual(values = c("#ECA400", "#C81D25", "#00A9A5", "green", "purple"), name = "Group") +
  labs(x = "UMAP1", y = "UMAP2", title = "Dependent") + theme(plot.title = element_text(hjust = 0.5))

set.seed(3)
umap <- umap(log10(useless_counts+1), init = "spectral", verbose = FALSE)
vis_umap = cbind(as.data.frame(umap), "Group" = as.factor(true_counts_res$cell_meta$pop))

useless = ggplot(vis_umap, aes(x = V1, y = V2)) + geom_point(aes(color = Group), alpha = 0.6, size = 1) +
  scale_color_manual(values = c("#ECA400", "#C81D25", "#00A9A5", "green", "purple"), name = "Group") +
  labs(x = "UMAP1", y = "UMAP2", title = "Noise") + theme(plot.title = element_text(hjust = 0.5))

set.seed(4)
umap <- umap(log10(joint_data+1), init = "spectral", verbose = FALSE)
vis_umap = cbind(as.data.frame(umap), "Group" = as.factor(true_counts_res$cell_meta$pop))

full = ggplot(vis_umap, aes(x = V1, y = V2)) + geom_point(aes(color = Group), alpha = 0.6, size = 1) +
  scale_color_manual(values = c("#ECA400", "#C81D25", "#00A9A5", "green", "purple"), name = "Group") +
  labs(x = "UMAP1", y = "UMAP2", title = "Combined") + theme(plot.title = element_text(hjust = 0.5))

plot_grid(causal, dep, useless, full, ncol=2)
ggsave('plots/220var_clean.pdf', height = 8, width = 10)

# save the dataset into data folder
print("Writing final dataset to data directory...")
cell_ids = vector()
for(i in 1:500) { cell_ids = append(cell_ids, paste(i, "_", true_counts_res$cell_meta$pop[i], sep = ""))}
rownames(joint_data) = cell_ids
write.csv(joint_data, '../../data/220var_clean.csv', row.names = TRUE)

# simulate the data with batch effects now 
print("Adding batch effects to the dataset...")
data(gene_len_pool)
ngenes = 2000
set.seed(1)
gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)
observed_counts <- DivideBatches(True2ObservedCounts(true_counts=true_counts_res[[1]], meta_cell=true_counts_res[[3]], protocol="UMI", alpha_mean=0.05, 
                                                     alpha_sd=0.02, gene_len=gene_len, depth_mean=10e6, depth_sd=3e6), nbatch = 2, batch_effect_size = 4)

# select the same 10 "causal" genes that are most DE between all 3 groups
print("Selecting the same 20 causal genes...")
causal_genes = (DE_genes_table %>% filter(DE == 'yes') %>% group_by(gene_short_name) %>% 
                  dplyr::mutate(n_DE = n(),
                                mean_absLFC = mean(abs(LFC_theoretical))) %>% 
                  ungroup() %>%
                  dplyr::select(gene_short_name, mean_absLFC) %>% distinct() %>%
                  slice_max(order_by=mean_absLFC, n=20))$gene_short_name
causal_genes_idx = as.numeric(gsub('gene', '', causal_genes))
causal_counts = t(observed_counts$counts) %>% as.data.frame() %>% dplyr::select(all_of(causal_genes_idx))
colnames(causal_counts) = gsub("V", "causal", colnames(causal_counts))

# randomly select the same 100 genes that are not DE in any group comparison
print("Selecting the same 100 noise genes...")
set.seed(66)
useless_genes = sample((DE_genes_table %>% filter(DE == 'no') %>% group_by(gene_short_name) %>% 
                          dplyr::mutate(n_DE = n()) %>% ungroup() %>% filter(n_DE == 20) %>%
                          dplyr::select(gene_short_name) %>% distinct())$gene_short_name, 100)
useless_genes_idx = as.numeric(gsub('gene', '', useless_genes))
useless_counts = t(observed_counts$counts) %>% as.data.frame() %>% dplyr::select(all_of(useless_genes_idx))
colnames(useless_counts) = gsub("V", "useless", colnames(useless_counts))

print("Simulating 100 variables that are linear combinations of the causal genes...")
set.seed(2020)
dep_counts = generateDepVariables(causal_counts, 100)

# join to obtain the final dataset
joint_data = cbind(causal_counts, useless_counts, dep_counts)

# plotting
print("Plotting data...")

set.seed(1)
umap <- umap(log10(causal_counts+1), init = "spectral", verbose = FALSE)
vis_umap = cbind(as.data.frame(umap), "Group" = as.factor(observed_counts$cell_meta$pop))

causal = ggplot(vis_umap, aes(x = V1, y = V2)) + geom_point(aes(color = Group), alpha = 0.6, size = 1) +
  scale_color_manual(values = c("#ECA400", "#C81D25", "#00A9A5", "green", "purple"), name = "Group") +
  labs(x = "UMAP1", y = "UMAP2", title = "Causal") + theme(plot.title = element_text(hjust = 0.5))

set.seed(2)
umap <- umap(log10(dep_counts+1), init = "spectral", verbose = FALSE)
vis_umap = cbind(as.data.frame(umap), "Group" = as.factor(observed_counts$cell_meta$pop))

dep = ggplot(vis_umap, aes(x = V1, y = V2)) + geom_point(aes(color = Group), alpha = 0.6, size = 1) +
  scale_color_manual(values = c("#ECA400", "#C81D25", "#00A9A5", "green", "purple"), name = "Group") +
  labs(x = "UMAP1", y = "UMAP2", title = "Dependent") + theme(plot.title = element_text(hjust = 0.5))

set.seed(3)
umap <- umap(log10(useless_counts+1), init = "spectral", verbose = FALSE)
vis_umap = cbind(as.data.frame(umap), "Group" = as.factor(observed_counts$cell_meta$pop))

useless = ggplot(vis_umap, aes(x = V1, y = V2)) + geom_point(aes(color = Group), alpha = 0.6, size = 1) +
  scale_color_manual(values = c("#ECA400", "#C81D25", "#00A9A5", "green", "purple"), name = "Group") +
  labs(x = "UMAP1", y = "UMAP2", title = "Noise") + theme(plot.title = element_text(hjust = 0.5))

set.seed(4)
umap <- umap(log10(joint_data+1), init = "spectral", verbose = FALSE)
vis_umap = cbind(as.data.frame(umap), "Group" = as.factor(observed_counts$cell_meta$pop))

full = ggplot(vis_umap, aes(x = V1, y = V2)) + geom_point(aes(color = Group), alpha = 0.6, size = 1) +
  scale_color_manual(values = c("#ECA400", "#C81D25", "#00A9A5", "green", "purple"), name = "Group") +
  labs(x = "UMAP1", y = "UMAP2", title = "Combined") + theme(plot.title = element_text(hjust = 0.5))
plot_grid(causal, dep, useless, full, ncol=2)
ggsave('plots/220var_hard.pdf', height = 8, width = 10)

# save the dataset into data folder
print("Writing final dataset to data directory...")
cell_ids = vector()
for(i in 1:500) { cell_ids = append(cell_ids, paste(i, "_", observed_counts$cell_meta$pop[i], "_", observed_counts$cell_meta$batch[i], sep = ""))}
rownames(joint_data) = cell_ids
write.csv(joint_data, '../../data/220var_hard.csv', row.names = TRUE)

print("Done.")
