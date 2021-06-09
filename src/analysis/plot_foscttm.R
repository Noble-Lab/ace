library(tidyverse)
require(cowplot)
library(cluster)
theme_set(theme_cowplot())

setwd('~/Documents/Github/2020_ylu465_group_explain/src/analysis')

# baseline silhouette:
raw_data = read.csv('../../data/220var_clean.csv', header = TRUE, row.names = 1)
metadata = data.frame("unique_id" = rownames(raw_data)) %>% separate(col = "unique_id", into = c("cellid", "cluster"), sep = "_")
data_subset = as.data.frame(log(raw_data+1)) %>% select(c(1:20))
sil = silhouette(as.numeric(metadata$cluster), daisy(data_subset))
mean_sil_score = mean(sil[,3])

# silhouette
deseq2 = read.csv('../../results/metrics_220var/deseq2_sil.csv', header = FALSE) %>% mutate(method = "deseq2")
grscore = read.csv('../../results/metrics_220var/grscore_sil.csv', header = FALSE) %>% mutate(method = "grscore")
js = read.csv('../../results/metrics_220var/jensen-shannon_sil.csv', header = FALSE) %>% mutate(method = "jensen-shannon")
#random = read.csv('../../results/metrics_220var/random_sil.csv', header = FALSE) %>% mutate(method = "random")
gce = read.csv('../../results/metrics_220var/gce_sil.csv', header = FALSE) %>% mutate(method = "gce")
explainer1 = read.csv('../../exp/exp_metrics_220var/explainer_mar0.5_fd1_latent2_sil.csv', header = FALSE) %>% mutate(method = "explainer_mar0.5_fd1_latent2")
explainer2 = read.csv('../../exp/exp_metrics_220var/explainer_mar0.5_fd1_latent10_sil.csv', header = FALSE) %>% mutate(method = "explainer_mar0.5_fd1_latent10")
explainer3 = read.csv('../../exp/exp_metrics_220var/explainer_mar0.5_fd50_latent10_sil.csv', header = FALSE) %>% mutate(method = "explainer_mar0.5_fd50_latent10")
explainer4 = read.csv('../../exp/exp_metrics_220var/explainer_mar0.5_fd50_latent2_sil.csv', header = FALSE) %>% mutate(method = "explainer_mar0.5_fd50_latent2")
explainer5 = read.csv('../../exp/exp_metrics_220var/explainer_mar0_fd1_latent2_sil.csv', header = FALSE) %>% mutate(method = "explainer_mar0.0_fd1_latent2")
explainer6 = read.csv('../../exp/exp_metrics_220var/explainer_mar0_fd1_latent10_sil.csv', header = FALSE) %>% mutate(method = "explainer_mar0.0_fd1_latent10")
explainer7 = read.csv('../../exp/exp_metrics_220var/explainer_mar0_fd50_latent10_sil.csv', header = FALSE) %>% mutate(method = "explainer_mar0.0_fd50_latent10")
explainer8 = read.csv('../../exp/exp_metrics_220var/explainer_mar0_fd50_latent2_sil.csv', header = FALSE) %>% mutate(method = "explainer_mar0.0_fd50_latent2")

# plotting
data = rbind(gce, deseq2, explainer7, js, grscore)
#data = rbind(explainer1, explainer2, explainer3, explainer4, explainer5, explainer6, explainer7, explainer8)
a = data %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Silhouette Score")
data %>% filter(V1 <= 20) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Silhouette Score")
plot_grid(a,b, ncol = 2)

# foscttm
deseq2 = read.csv('../../results/metrics_220var/deseq2_foscttm.csv', header = FALSE) %>% mutate(method = "deseq2")
grscore = read.csv('../../results/metrics_220var/grscore_foscttm.csv', header = FALSE) %>% mutate(method = "grscore")
js = read.csv('../../results/metrics_220var/jensen-shannon_foscttm.csv', header = FALSE) %>% mutate(method = "jensen-shannon")
#random = read.csv('../../results/metrics_220var/random_jaccard.csv', header = FALSE) %>% mutate(method = "random")
gce = read.csv('../../results/metrics_220var/gce_jaccard.csv', header = FALSE) %>% mutate(method = "gce")

explainer1 = read.csv('../../exp/exp_metrics_220var/explainer_mar0.5_fd1_latent2_foscttm.csv', header = FALSE) %>% mutate(method = "explainer_mar0.5_fd1_latent2")
explainer2 = read.csv('../../exp/exp_metrics_220var/explainer_mar0.5_fd1_latent10_foscttm.csv', header = FALSE) %>% mutate(method = "explainer_mar0.5_fd1_latent10")
explainer3 = read.csv('../../exp/exp_metrics_220var/explainer_mar0.5_fd50_latent10_foscttm.csv', header = FALSE) %>% mutate(method = "explainer_mar0.5_fd50_latent10")
explainer4 = read.csv('../../exp/exp_metrics_220var/explainer_mar0.5_fd50_latent2_foscttm.csv', header = FALSE) %>% mutate(method = "explainer_mar0.5_fd50_latent2")
explainer5 = read.csv('../../exp/exp_metrics_220var/explainer_mar0_fd1_latent2_foscttm.csv', header = FALSE) %>% mutate(method = "explainer_mar0.0_fd1_latent2")
explainer6 = read.csv('../../exp/exp_metrics_220var/explainer_mar0_fd1_latent10_foscttm.csv', header = FALSE) %>% mutate(method = "explainer_mar0.0_fd1_latent10")
explainer7 = read.csv('../../exp/exp_metrics_220var/explainer_mar0_fd50_latent10_foscttm.csv', header = FALSE) %>% mutate(method = "explainer_mar0.0_fd50_latent10")
explainer8 = read.csv('../../exp/exp_metrics_220var/explainer_mar0_fd50_latent2_foscttm.csv', header = FALSE) %>% mutate(method = "explainer_mar0.0_fd50_latent2")

# plotting
#data = rbind(gce, deseq2, random, js, grscore)
data = rbind(explainer1, explainer2, explainer3, explainer4, explainer5, explainer6, explainer7, explainer8)
a = data %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean FOSCTTM")
b = data %>% filter(V1 <= 50) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean FOSCTTM")
plot_grid(a,b, ncol = 2)

# jaccard
deseq2 = read.csv('../../results/metrics_220var/deseq2_jaccard.csv', header = FALSE) %>% mutate(method = "DESeq2")
grscore = read.csv('../../results/metrics_220var/grscore_jaccard.csv', header = FALSE) %>% mutate(method = "GRScore")
js = read.csv('../../results/metrics_220var/jensen-shannon_jaccard.csv', header = FALSE) %>% mutate(method = "JSD")
#random = read.csv('../../results/metrics_220var/random_jaccard.csv', header = FALSE) %>% mutate(method = "random")
gce = read.csv('../../results/metrics_220var/gce_jaccard.csv', header = FALSE) %>% mutate(method = "GCE")
explainer1 = read.csv('../../exp/exp_metrics_220var/explainer_mar0.5_fd1_latent2_jaccard.csv', header = FALSE) %>% mutate(method = "explainer_mar0.5_fd1_latent2")
explainer2 = read.csv('../../exp/exp_metrics_220var/explainer_mar0.5_fd1_latent10_jaccard.csv', header = FALSE) %>% mutate(method = "explainer_mar0.5_fd1_latent10")
explainer3 = read.csv('../../exp/exp_metrics_220var/explainer_mar0.5_fd50_latent10_jaccard.csv', header = FALSE) %>% mutate(method = "ACE")
explainer4 = read.csv('../../exp/exp_metrics_220var/explainer_mar0.5_fd50_latent2_jaccard.csv', header = FALSE) %>% mutate(method = "explainer_mar0.5_fd50_latent2")
explainer5 = read.csv('../../exp/exp_metrics_220var/explainer_mar0_fd1_latent2_jaccard.csv', header = FALSE) %>% mutate(method = "explainer_mar0.0_fd1_latent2")
explainer6 = read.csv('../../exp/exp_metrics_220var/explainer_mar0_fd1_latent10_jaccard.csv', header = FALSE) %>% mutate(method = "explainer_mar0.0_fd1_latent10")
explainer7 = read.csv('../../exp/exp_metrics_220var/explainer_mar0_fd50_latent10_jaccard.csv', header = FALSE) %>% mutate(method = "explainer_mar0.0_fd50_latent10")
explainer8 = read.csv('../../exp/exp_metrics_220var/explainer_mar0_fd50_latent2_jaccard.csv', header = FALSE) %>% mutate(method = "explainer_mar0.0_fd50_latent2")

# plotting
data = rbind(explainer1, explainer2, explainer3, explainer4, explainer5, explainer6, explainer7, explainer8)
data = rbind(deseq2, grscore, js, gce, explainer3)

a = data %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Jaccard Distance")
data %>% filter(V1 <= 30) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Jaccard Distance")
plot_grid(a,b, ncol = 2)

data %>% filter(V1 <= 30) %>% ggplot(aes(x = V1, y = V2)) + geom_hline(yintercept=0.44318, linetype = 'dashed', color = 'grey30', size = 1) +
  geom_line(aes(color = method), size = 1) +
  scale_color_manual(values = c("#F02D3A", "#7FB800", "#F39237", "#81559B", "#2E86AB")) +
  labs(x = "Number of genes used", y = "Mean Jaccard Distance")

data %>% filter(V1 <= 30, method == 'HAHA') %>% ggplot(aes(x = V1, y = V2)) + geom_hline(yintercept=0.44318, linetype = 'dashed', color = 'grey30', size = 1) +
  geom_line(aes(color = method), size = 1) +
  labs(x = "Number of genes used", y = "Mean Jaccard Distance") 
ggsave('~/Desktop/result.pdf', height = 4, width = 5.5)
