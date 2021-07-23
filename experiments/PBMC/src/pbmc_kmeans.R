suppressMessages(library(tidyverse))
require(cowplot)
theme_set(theme_cowplot())

setwd("~/Desktop/PBMC/src")

umap = read.table('../data/3K_PBMC_umap_embedding.txt', header = FALSE, sep = ',')
leiden_labels = read.table('../data/3K_PBMC_clust_ids_leiden.txt', header = FALSE)$V1
leiden_labels = recode(leiden_labels, 
       `0` = "CD4 T cells", 
       `1` = "CD14+ Monocytes", 
       `2` = "B cells",
       `3` = "CD8 T cells", 
       `4` = "NK cells", 
       `5` = "FCGR3A+ Monocytes", 
       `6` = "Dendritic Cells", 
       `7` = "Megakaryocytes")

kmeans_labels = read.table('../data/3K_PBMC_clust_ids_kmeans.txt', header = FALSE)$V1
kmeans_labels = recode(kmeans_labels, 
        `0` = "CD4 T cells", 
        `1` = "CD14+ Monocytes", 
        `2` = "B cells",
        `3` = "CD8 T cells", 
        `4` = "NK cells", 
        `5` = "FCGR3A+ Monocytes", 
        `6` = "Dendritic Cells", 
        `7` = "Megakaryocytes")

p1 <- umap %>% ggplot(aes(x = V1, y = V2)) + geom_point(aes(color = as.factor(leiden_labels)), size = 0.75, alpha = 0.8) + 
  labs(x = 'UMAP1', y = 'UMAP2', title = 'Leiden clustering') +
  theme(legend.title = element_blank())
p2 <- umap %>% ggplot(aes(x = V1, y = V2)) + geom_point(aes(color = as.factor(kmeans_labels)), size = 0.75, alpha = 0.8) + 
  labs(x = 'UMAP1', y = 'UMAP2', title = 'k-means clustering') +
  theme(legend.title = element_blank())
plot_grid(p1, p2, labels = "AUTO")
ggsave('../figs/kmeans_PBMC.pdf', height = 5, width = 14)
