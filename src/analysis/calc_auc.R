library(tidyverse)
library(ROCR)
require(cowplot)
theme_set(theme_cowplot())

setwd('~/Documents/Github/2020_ylu465_group_explain/src/analysis')

# true labels
true_labels = data.frame("gene_short_name" = colnames(read.csv('../../data/140var_clean.csv', header = TRUE, row.names = 1))) %>%
  mutate(DE = ifelse(grepl("causal|dep", .$gene_short_name), "yes", "no"))

# results files
js.res = read.csv('../../results/method_output_220var/jensen_shannon_result.csv', header = TRUE) 
deseq.res = read.csv('../../results/method_output_140var/deseq2_result.csv', header = TRUE) %>% mutate(padj = ifelse(is.na(padj), 1, padj))
grscore.res = read.csv('../../results/method_output/grscore_result.csv', header = TRUE) %>% mutate(global_relevance = -global_relevance)
gce.res = read.csv('../../results/method_output_140var/gce_result.csv', header = FALSE) %>% cbind(true_labels$gene_short_name) %>%
  dplyr::rename(relevance = `V1`, gene_short_name = `true_labels$gene_short_name`)
explainer.res = read.csv('../../results/method_output_220var/figs_220var_hard_beta0.08_margin0_fd-1_latent2/explainer_result.csv', header = FALSE) %>% cbind(true_labels$gene_short_name) %>%
  dplyr::rename(relevance = `V1`, gene_short_name = `true_labels$gene_short_name`)


# AUC curves
result = data.frame("Method" = NULL, "TPR" = NULL, "FPR" = NULL, "AUC" = NULL, stringsAsFactors = FALSE)

pred = prediction(-deseq.res$padj, true_labels$DE, label.ordering = c("no", "yes"))
perf = performance(pred,"tpr","fpr")
auc = performance(pred, "auc")@y.values[[1]]
result = rbind(result, data.frame("Method" = "deseq2",  "TPR" = unlist(perf@y.values), "FPR" = unlist(perf@x.values), 
                    "AUC" = auc, stringsAsFactors = FALSE))

pred = prediction(js.res$q_value, filter(true_labels, gene_short_name %in% js.res$gene_short_name)$DE, label.ordering = c("yes", "no"))
perf = performance(pred,"fpr","tpr")
auc = performance(pred, "auc")@y.values[[1]]
result = rbind(result, data.frame("Method" = "jensen-shannon",  "TPR" = unlist(perf@y.values), "FPR" = unlist(perf@x.values), 
                    "AUC" = auc, stringsAsFactors = FALSE))

pred = prediction(grscore.res$global_relevance, filter(true_labels, gene_short_name %in% grscore.res$genes)$DE, label.ordering = c("yes", "no"))
perf = performance(pred,"tpr","fpr")
auc = performance(pred, "auc")@y.values[[1]]
result = rbind(result, data.frame("Method" = "gene relevance score",  "TPR" = unlist(perf@y.values), "FPR" = unlist(perf@x.values), 
                    "AUC" = auc, stringsAsFactors = FALSE))

pred = prediction(gce.res$relevance, true_labels$DE, label.ordering = c("yes", "no"))
perf = performance(pred,"tpr","fpr")
auc = performance(pred, "auc")@y.values[[1]]
result = rbind(result, data.frame("Method" = "global counterfactual explanation",  "TPR" = unlist(perf@y.values), "FPR" = unlist(perf@x.values), 
                                  "AUC" = auc, stringsAsFactors = FALSE))

pred = prediction(explainer.res$relevance, true_labels$DE, label.ordering = c("yes", "no"))
perf = performance(pred,"tpr","fpr")
auc = performance(pred, "auc")@y.values[[1]]
result = rbind(result, data.frame("Method" = "explainer",  "TPR" = unlist(perf@y.values), "FPR" = unlist(perf@x.values), 
                    "AUC" = auc, stringsAsFactors = FALSE))

# plot
result %>% ggplot(aes(x = FPR, y = TPR)) + geom_line(aes(color = Method), size = 1) + 
  geom_abline(slope = 1, intercept = 0, linetype = 'dashed', color = 'black', size = 0.75) +
  labs(x = 'False Positive Rate', y = 'True Positive Rate')

# =========
library(viridis)

js.res = read.csv('../../results/method_output_140var/jensen_shannon_result.csv', header = TRUE) 
deseq.res = read.csv('../../results/method_output_140var/deseq2_result.csv', header = TRUE) %>% mutate(padj = ifelse(is.na(padj), 1, padj))
gce.res = read.csv('../../results/method_output_140var/gce_result.csv', header = FALSE) %>% cbind(true_labels$gene_short_name) %>%
  dplyr::rename(relevance = `V1`, gene_short_name = `true_labels$gene_short_name`)
explainer.res = read.csv('../../results/method_output_140var/figs_140var_clean_beta0.01_margin0.3_fd20_latent2/explainer_result.csv', header = FALSE) %>% cbind(true_labels$gene_short_name) %>%
  dplyr::rename(relevance = `V1`, gene_short_name = `true_labels$gene_short_name`)
grscore.res = read.csv('../../results/method_output_140var/grscore_result.csv', header = TRUE)

minmax <- function(x){(x-min(x))/(max(x)-min(x))}
deseq.res = deseq.res %>% mutate(importance = minmax(-scale(padj)), method = "DESeq2") %>% select(-padj)
js.res = js.res %>% mutate(importance = minmax(-scale(q_value)), method = "Jensen-Shannon Distance") %>% select(-q_value)
gce.res = gce.res %>% mutate(importance = minmax(scale(relevance)), method = "Global Counterfactual Explanation") %>% select(-relevance)
explainer.res = explainer.res %>% mutate(importance = minmax(scale(relevance)), method = "Explainer") %>% select(-relevance)
grscore.res = grscore.res %>% mutate(gene_short_name = genes, importance = minmax(scale(local_relevance)), method = "Gene Relevance Score") %>% select(-c(local_relevance, genes))

result = rbind(deseq.res, gce.res, js.res, explainer.res, grscore.res)

result %>% ggplot(aes(x = gene_short_name, y = method)) + geom_tile(aes(fill = importance), alpha = 0.9, color = 'black', size = 0.3) +
  scale_fill_viridis() +
  scale_x_discrete(limits = true_labels$gene_short_name) +
  scale_y_discrete(limits = c("DESeq2", "Jensen-Shannon Distance", "Gene Relevance Score", "Global Counterfactual Explanation", "Explainer")) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

sparse.sol = explainer.res %>% top_n(n=, wt=importance)
sparse.sol = gce.res %>% top_n(n=5, wt=importance)
sparse.sol = deseq.res %>% top_n(n=5, wt=importance)

raw = read.csv('../../data/140var_clean.csv', header = TRUE, row.names = 1)
metadata = data.frame("unique_id" = rownames(raw)) %>% separate(col = "unique_id", into = c("cellid", "cluster"), sep = "_")

library(uwot)

set.seed(1)
umap <- umap(scale(log((raw %>% select(sparse.sol$gene_short_name))+1)), init = "spectral", verbose = FALSE)
vis_umap = cbind(as.data.frame(umap), "Group" = as.factor(metadata$cluster))

ggplot(vis_umap, aes(x = V1, y = V2)) + geom_point(aes(color = Group), alpha = 0.6, size = 1) +
  scale_color_manual(values = c("#ECA400", "#C81D25", "#00A9A5", "green", "purple"), name = "Group") +
  labs(x = "UMAP1", y = "UMAP2", title = "7 top genes") + theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size= 20),
        axis.text.x = element_text(size =15),
        axis.text.y = element_text(size = 15))

