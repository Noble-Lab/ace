library(tidyverse)
require(cowplot)
theme_set(theme_cowplot())

setwd('~/Desktop/simulation/results/metrics/')

# CLEAN: ONE VS. REST
# FOSCTTM ---
deseq2 = read.csv('deseq2_one_vs_rest_clean_foscttm.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grscore_one_vs_rest_clean_foscttm.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jensen-shannon_one_vs_rest_clean_foscttm.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_one_vs_rest_clean_foscttm.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_one_vs_rest_clean_foscttm.csv', header = FALSE) %>% mutate(method = "ACE")
data = rbind(deseq2, grs, jsd, gce, ace)
data %>% filter(V1 <= 40) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean FOSCTTM")

# Silhouette ---
deseq2 = read.csv('deseq2_one_vs_rest_clean_sil.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grscore_one_vs_rest_clean_sil.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jensen-shannon_one_vs_rest_clean_sil.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_one_vs_rest_clean_sil.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_one_vs_rest_clean_sil.csv', header = FALSE) %>% mutate(method = "ACE")
data = rbind(deseq2, grs, jsd, gce, ace)
data %>% filter(V1 <= 35) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Silhouette Score")

# Jaccard ---
deseq2 = read.csv('deseq2_clean_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grs_clean_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jsd_clean_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_clean_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_clean_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "ACE")
data = rbind(deseq2, grs, jsd, ace, gce)
data %>% filter(V1 <= 50) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Jaccard Distance")

# CLEAN: ONE VS. ONE
# FOSCTTM ---
deseq2 = read.csv('deseq2_one_vs_one_clean_foscttm.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grscore_one_vs_one_clean_foscttm.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jensen-shannon_one_vs_one_clean_foscttm.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_one_vs_one_clean_foscttm.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_one_vs_one_clean_foscttm.csv', header = FALSE) %>% mutate(method = "ACE")
data = rbind(deseq2, grs, jsd, gce, ace)
data %>% filter(V1 <= 40) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean FOSCTTM")

# Silhouette ---
deseq2 = read.csv('deseq2_one_vs_one_clean_sil.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grscore_one_vs_one_clean_sil.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jensen-shannon_one_vs_one_clean_sil.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_one_vs_one_clean_sil.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_one_vs_one_clean_sil.csv', header = FALSE) %>% mutate(method = "ACE")
data = rbind(deseq2, grs, jsd, gce, ace)
data %>% filter(V1 <= 40) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Silhouette Score")

# Jaccard ---
deseq2 = read.csv('deseq2_one_vs_one_clean_jaccard.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grscore_one_vs_one_clean_jaccard.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jensen-shannon_one_vs_one_clean_jaccard.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_one_vs_one_clean_jaccard.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_one_vs_one_clean_jaccard.csv', header = FALSE) %>% mutate(method = "ACE")
data = rbind(deseq2, grs, jsd, gce, ace)
data %>% filter(V1 <= 40) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Jaccard Distance")

# HARD: ONE VS. REST
# FOSCTTM ---
deseq2 = read.csv('deseq2_one_vs_rest_hard_foscttm.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grscore_one_vs_rest_hard_foscttm.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jensen-shannon_one_vs_rest_hard_foscttm.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_one_vs_rest_hard_foscttm.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_one_vs_rest_hard_foscttm.csv', header = FALSE) %>% mutate(method = "ACE")
data = rbind(deseq2, grs, jsd, gce, ace)
data %>% filter(V1 <= 40) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean FOSCTTM")

# Silhouette ---
deseq2 = read.csv('deseq2_one_vs_rest_hard_sil.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grscore_one_vs_rest_hard_sil.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jensen-shannon_one_vs_rest_hard_sil.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_one_vs_rest_hard_sil.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_one_vs_rest_hard_sil.csv', header = FALSE) %>% mutate(method = "ACE")
data = rbind(deseq2, grs, jsd, gce, ace)
data %>% filter(V1 <= 35) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Silhouette Score")

# Jaccard ---
deseq2 = read.csv('deseq2_one_vs_rest_hard_jaccard.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grscore_one_vs_rest_hard_jaccard.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jensen-shannon_one_vs_rest_hard_jaccard.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_one_vs_rest_hard_jaccard.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_one_vs_rest_hard_jaccard.csv', header = FALSE) %>% mutate(method = "ACE")
data = rbind(deseq2, grs, jsd, gce, ace)
data %>% filter(V1 <= 40) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Jaccard Distance")

# HARD: ONE VS. ONE
# FOSCTTM ---
deseq2 = read.csv('deseq2_one_vs_one_hard_foscttm.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grscore_one_vs_one_hard_foscttm.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jensen-shannon_one_vs_one_hard_foscttm.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_one_vs_one_hard_foscttm.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_one_vs_one_hard_foscttm.csv', header = FALSE) %>% mutate(method = "ACE")
data = rbind(deseq2, grs, jsd, gce, ace)
data %>% filter(V1 <= 40) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean FOSCTTM")

# Silhouette ---
deseq2 = read.csv('deseq2_one_vs_one_hard_sil.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grscore_one_vs_one_hard_sil.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jensen-shannon_one_vs_one_hard_sil.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_one_vs_one_hard_sil.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_one_vs_one_hard_sil.csv', header = FALSE) %>% mutate(method = "ACE")
data = rbind(deseq2, grs, jsd, gce, ace)
data %>% filter(V1 <= 35) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Silhouette Score")

# Jaccard ---
deseq2 = read.csv('deseq2_one_vs_one_hard_jaccard.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grscore_one_vs_one_hard_jaccard.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jensen-shannon_one_vs_one_hard_jaccard.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_one_vs_one_hard_jaccard.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_one_vs_one_hard_jaccard.csv', header = FALSE) %>% mutate(method = "ACE")
data = rbind(deseq2, grs, jsd, gce, ace)
data %>% filter(V1 <= 40) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Jaccard Distance")


# ==== ACTUAL FIGURE ====
deseq2 = read.csv('deseq2_clean_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grs_clean_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jsd_clean_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_clean_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_clean_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "ACE")
sal = read.csv('sal_clean_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "SmoothGrad")
shap = read.csv('shap_clean_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "SHAP")

data = rbind(deseq2, grs, jsd, gce, sal, shap, ace)
data$method <- factor(data$method, c("GCE", "GRS", "JSD", "DESeq2", "SmoothGrad", "SHAP", "ACE"))
clean = data %>% filter(V1 <= 50) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Jaccard Distance", title = "Clean") + 
  scale_color_manual(values = c("#7FB800", "#F39237", "#ffa2b0", "#81559B", "#2E86AB", "#000080", "#F02D3A")) +
  geom_hline(yintercept = 0.5387006286880948, linetype = 'dashed', color = 'grey30', size = 0.75)

deseq2 = read.csv('deseq2_hard_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "DESeq2")
grs = read.csv('grs_hard_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "GRS")
jsd = read.csv('jsd_hard_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "JSD")
gce = read.csv('gce_hard_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "GCE")
ace = read.csv('ace_hard_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "ACE")
sal = read.csv('sal_hard_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "SmoothGrad")
shap = read.csv('shap_hard_one_vs_rest_jaccard.csv', header = FALSE) %>% mutate(method = "SHAP")

data = rbind(deseq2, grs, jsd, gce, sal, shap, ace)
data$method <- factor(data$method, c("GCE", "GRS", "JSD", "DESeq2", "SmoothGrad", "SHAP", "ACE"))
hard = data %>% filter(V1 <= 50) %>% ggplot(aes(x = V1, y = V2)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of genes used", y = "Mean Jaccard Distance", title = "Complex") + 
  scale_color_manual(values = c("#7FB800", "#F39237", "#ffa2b0", "#81559B", "#2E86AB", "#000080", "#F02D3A")) +
  geom_hline(yintercept = 0.5387006286880948, linetype = 'dashed', color = 'grey30', size = 0.75)

plot_grid(clean, hard, ncol=2, labels="AUTO")
ggsave('../../final_figures/performance.pdf', height = 4, width = 10)
