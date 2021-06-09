library(cluster)
library(tidyverse)
#library(aricode)
require(cowplot)
theme_set(theme_cowplot())

setwd('~/Documents/Github/2020_ylu465_group_explain/src/analysis')

raw_data = read.csv('../../data/220var_clean.csv', header = TRUE, row.names = 1)
metadata = data.frame("unique_id" = rownames(raw_data)) %>% separate(col = "unique_id", into = c("cellid", "cluster"), sep = "_")

# jensen-shannon result
js.res = read.csv('../../results/method_output_220var/jensen_shannon_result.csv', header = TRUE)
js.ranking = unique(js.res[order(js.res$q_value),]$gene_short_name) # order by significance

# grscore result
grscore.res = read.csv('../../results/method_output_220var/grscore_result.csv', header = TRUE)
grscore.ranking = unique(grscore.res[order(-grscore.res$local_relevance),]$genes) # order by significance

# deseq2 result
deseq.res = read.csv('../../results/method_output_220var/deseq2_result.csv', header = TRUE)
deseq.ranking = unique(deseq.res[order(deseq.res$padj),]$gene_short_name) # order by significance

# GCE result
gce.res = read.csv('../../results/method_output_220var/gce_result.csv', header = FALSE) %>% cbind(colnames(raw_data)) %>% 
  dplyr::rename(relevance = `V1`, gene_short_name = `colnames(raw_data)`)
gce.ranking = unique(deseq.res[order(-gce.res$relevance),]$gene_short_name) # order by relevance

# explainer 1 result
explainer.res_mar0_fd1_latent2 = read.csv('../../results/method_output_220var/figs_220var_clean_beta0.01_margin0_fd-1_latent2/explainer_result.csv', header = FALSE) %>% cbind(colnames(raw_data)) %>%
  dplyr::rename(relevance = `V1`, gene_short_name = `colnames(raw_data)`) 
explainer_mar0_fd1_latent2.ranking = unique(explainer.res_mar0_fd1_latent2[order(-explainer.res_mar0_fd1_latent2$relevance),]$gene_short_name) # order by relevance

# explainer 2 result
explainer.res_mar0_fd25_latent2 = read.csv('../../results/method_output_220var/figs_220ar_clean_beta0.01_margin0_fd25_latent2/explainer_result.csv', header = FALSE) %>% cbind(colnames(raw_data)) %>%
  dplyr::rename(relevance = `V1`, gene_short_name = `colnames(raw_data)`) 
explainer_mar0_fd25_latent2.ranking = unique(explainer.res_mar0_fd25_latent2[order(-explainer.res_mar0_fd25_latent2$relevance),]$gene_short_name) # order by relevance

# explainer 3 result
explainer.res_mar0_fd1_latent10 = read.csv('../../results/method_output_220var/figs_220var_clean_beta0.01_margin0_fd-1_latent10/explainer_result.csv', header = FALSE) %>% cbind(colnames(raw_data)) %>%
  dplyr::rename(relevance = `V1`, gene_short_name = `colnames(raw_data)`) 
explainer_mar0_fd1_latent10.ranking = unique(explainer.res_mar0_fd1_latent10[order(-explainer.res_mar0_fd1_latent10$relevance),]$gene_short_name) # order by relevance

# explainer 4 result
explainer.res_mar0.25_fd1_latent2 = read.csv('../../results/method_output_220var/figs_220var_clean_beta0.01_margin0.5_fd100_latent10/explainer_result.csv', header = FALSE) %>% cbind(colnames(raw_data)) %>%
  dplyr::rename(relevance = `V1`, gene_short_name = `colnames(raw_data)`) 
explainer_mar0.25_fd1_latent2.ranking = unique(explainer.res_mar0.25_fd1_latent2[order(-explainer.res_mar0.25_fd1_latent2$relevance),]$gene_short_name) # order by relevance

# random ranking
set.seed(101)
random.ranking = sample(colnames(raw_data), 220)

cosineDistance <- function(feat_matrix) {
  M = as.matrix(feat_matrix)
  sim = M / sqrt(rowSums(M * M))
  sim = sim %*% t(sim)
  dist = as.dist(1 - sim)
  return(dist)
}

minmax <- function(x){(x-min(x))/(max(x)-min(x))}

evaluateSilhouette <- function(true_clusters, deseq_ranking, gr_ranking, js_ranking, gce_ranking, explain1_ranking, 
                               explain2_ranking, explain3_ranking, explain4_ranking, rand_ranking, raw_dataframe, n_feat_total) {
  res = data.frame("method" = NULL, "n_feat" = NULL, "sil_score" = NULL, stringsAsFactors = FALSE)
  for(i in 2:n_feat_total) {
      #DESeq2
      data_subset = as.data.frame(minmax(log10(raw_dataframe+1))) %>% select(deseq_ranking[1:i])
      dist_matrix = daisy(data_subset)
      sil = silhouette(as.numeric(true_clusters), dist_matrix)
      mean_sil_score = mean(sil[,3])
      res = rbind(res, data.frame("method"="DESeq2", "n_feat"=i, "sil_score"=mean_sil_score))

      # grscore
      data_subset = as.data.frame(minmax(log10(raw_dataframe+1))) %>% select(gr_ranking[1:i])
      dist_matrix = daisy(data_subset)
      sil = silhouette(as.numeric(true_clusters), dist_matrix)
      mean_sil_score = mean(sil[,3])
      res = rbind(res, data.frame("method"="Local Gene Relevance", "n_feat"=i, "sil_score"=mean_sil_score))
      
      # JS
      data_subset = as.data.frame(minmax(log10(raw_dataframe+1))) %>% select(js_ranking[1:i])
      dist_matrix = daisy(data_subset)
      sil = silhouette(as.numeric(true_clusters), dist_matrix)
      mean_sil_score = mean(sil[,3])
      res = rbind(res, data.frame("method"="Jensen-Shannon Distance", "n_feat"=i, "sil_score"=mean_sil_score))

      # GCE
      data_subset = as.data.frame(minmax(log10(raw_dataframe+1))) %>% select(gce_ranking[1:i])
      dist_matrix = daisy(data_subset)
      sil = silhouette(as.numeric(true_clusters), dist_matrix)
      mean_sil_score = mean(sil[,3])
      res = rbind(res, data.frame("method"="Global Counterfactual Explanation", "n_feat"=i, "sil_score"=mean_sil_score))

      # Explainer 1
      data_subset = as.data.frame(minmax(log10(raw_dataframe+1))) %>% select(explain1_ranking[1:i])
      dist_matrix = daisy(data_subset)
      sil = silhouette(as.numeric(true_clusters), dist_matrix)
      mean_sil_score = mean(sil[,3])
      res = rbind(res, data.frame("method"="Explainer_mar0_fd1_latent2", "n_feat"=i, "sil_score"=mean_sil_score))

      # Explainer 2
      data_subset = as.data.frame(minmax(log10(raw_dataframe+1))) %>% select(explain2_ranking[1:i])
      dist_matrix = daisy(data_subset)
      sil = silhouette(as.numeric(true_clusters), dist_matrix)
      mean_sil_score = mean(sil[,3])
      res = rbind(res, data.frame("method"="Explainer_mar0_fd25_latent2", "n_feat"=i, "sil_score"=mean_sil_score))

      # Explainer 3
      data_subset = as.data.frame(minmax(log10(raw_dataframe+1))) %>% select(explain3_ranking[1:i])
      dist_matrix = daisy(data_subset)
      sil = silhouette(as.numeric(true_clusters), dist_matrix)
      mean_sil_score = mean(sil[,3])
      res = rbind(res, data.frame("method"="Explainer_mar0_fd1_latent10", "n_feat"=i, "sil_score"=mean_sil_score))

      # Explainer 4
      data_subset = as.data.frame(minmax(log10(raw_dataframe+1))) %>% select(explain4_ranking[1:i])
      dist_matrix = daisy(data_subset)
      sil = silhouette(as.numeric(true_clusters), dist_matrix)
      mean_sil_score = mean(sil[,3])
      res = rbind(res, data.frame("method"="Explainer_mar0.25_fd1_latent2", "n_feat"=i, "sil_score"=mean_sil_score))

      # random guessing
      data_subset = as.data.frame(minmax(log10(raw_dataframe+1))) %>% select(rand_ranking[1:i])
      dist_matrix = daisy(data_subset)
      sil = silhouette(as.numeric(true_clusters), dist_matrix)
      mean_sil_score = mean(sil[,3])
      res = rbind(res, data.frame("method"="Random", "n_feat"=i, "sil_score"=mean_sil_score))

  }
  return(res)
}

set.seed(123)
test = evaluateSilhouette(metadata$cluster, deseq.ranking, grscore.ranking, js.ranking, gce.ranking, 
                          explainer_mar0_fd1_latent2.ranking, explainer_mar0_fd25_latent2.ranking,
                          explainer_mar0_fd1_latent10.ranking, explainer_mar0.25_fd1_latent2.ranking,
                          random.ranking, raw_data, 220)

a = test %>% group_by(method, n_feat) %>%
  ggplot(aes(x = n_feat, y = sil_score)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of marker genes", y = "Mean Silhouette Score")

b = test %>% filter(n_feat <= 50) %>% group_by(method, n_feat) %>%
  ggplot(aes(x = n_feat, y = sil_score)) + geom_line(aes(color = method), size = 1) + 
  labs(x = "Number of marker genes", y = "Mean Silhouette Score")

plot_grid(a,b, ncol=2)
# =======================================================================================================



ggplot(data = NULL, aes(x = raw_data$causal1907, y = raw_data$useless1271)) + geom_point()

#####
data_subset = as.data.frame(minmax(log(raw_data+1))) %>% select(explainer_mar0_fd25_latent2.ranking[1:2])
cosine.matrix = as.matrix(cosineDistance(data_subset))
sil = silhouette(as.numeric(metadata$cluster), cosine.matrix)
mean_sil_score = mean(sil[,3])

umap <- umap(log10((raw_data %>% select(colnames(data_subset)))+1), init = "spectral", verbose = FALSE)
vis_umap = cbind(as.data.frame(umap), "gene_exp" = log10(raw_data$causal420+1), "Group" = metadata$cluster)

ggplot(vis_umap, aes(x = V1, y = V2)) + geom_point(aes(color = gene_exp), alpha = 0.6, size = 2) +
  scale_color_viridis(name = "Gene Exp") +
  labs(x = "UMAP1", y = "UMAP2", title = "Cluster (easy)") + theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size= 20),
        axis.text.x = element_text(size =15),
        axis.text.y = element_text(size = 15))

ggplot(vis_umap, aes(x = V1, y = V2)) + geom_point(aes(color = as.factor(Group)), alpha = 0.6, size = 2) +
  scale_color_manual(values = c("#ECA400", "#C81D25", "#00A9A5", "green", "purple"), name = "Group") +
  labs(x = "UMAP1", y = "UMAP2", title = "Cluster (easy)") + theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size= 20),
        axis.text.x = element_text(size =15),
        axis.text.y = element_text(size = 15))




plot(sil)

test = prcomp(raw_data)
PC = as.data.frame(test$x)
ggplot(PC, aes(x = PC1, y = PC2)) + geom_point(aes(color = metadata$cluster))
library(ggfortify)

data = as.data.frame(cbind(metadata$cluster, raw_data %>% select(js.ranking[1:50])))
autoplot(prcomp(data[,-1], scale. = TRUE), data = data, colour = 'metadata$cluster', size = 1)

library(pROC)

multiclass.roc(metadata$cluster, data_subset$gene284)







# moran's result
morans.res = read.csv('../../results/morans_result.csv', header = TRUE)
morans.ranking = morans.res[order(-morans.res$morans_I),]$gene_short_name # order by moran's I

# gene relevance result
gr.res = read.csv('../../results/grscore_result.csv', header = TRUE)
gr.ranking = gr.res[order(-gr.res$global_relevance),]$genes # order by moran's I


evaluateSilhouette <- function(true_clusters, gene_ranking, raw_dataframe) {
  res = data.frame("n_feat" = NULL, "sil_score" = NULL, stringsAsFactors = FALSE)
  for(i in 0:length(gene_ranking)) {
    if(i == 0) {
      res = rbind(res, data.frame("n_feat"=0, "sil_score"=0))
    } else {
      data_subset = raw_data %>% select(gene_ranking[1:i])
      dist_matrix = daisy(data_subset, metric = "euclidean")
      sil = silhouette(as.numeric(true_clusters), dist_matrix)
      mean_sil_score = mean(sil[,3])
      res = rbind(res, data.frame("n_feat"=i, "sil_score"=mean_sil_score))
    }
  }
  return(res)
}
















evaluateAdjustedMutualInformation <- function(true_clusters, js_ranking, morans_ranking, gr_ranking, raw_dataframe, nrep) {
  res = data.frame("n_feat" = NULL, "AMI" = NULL, "method" = NULL, stringsAsFactors = FALSE)
  for(i in 2:50) {
    print(paste("number of markers: ", i))
    js_scores = vector()
    morans_scores = vector()
    gr_scores = vector()
    random_scores = vector()
    for(j in 1:nrep) {
      # k-means for actual selected markers
      predicted_js = as.vector(kmeans(log10((raw_data %>% select(js_ranking[1:i]))+1), centers=5)$cluster)
      predicted_morans = as.vector(kmeans(log10((raw_data %>% select(morans_ranking[1:i]))+1), centers=5)$cluster)
      predicted_gr = as.vector(kmeans(log10((raw_data %>% select(gr_ranking[1:i]))+1), centers=5)$cluster)
      js_scores = append(js_scores, AMI(true_clusters, predicted_js))
      morans_scores = append(morans_scores, AMI(true_clusters, predicted_morans))
      gr_scores = append(gr_scores, AMI(true_clusters, predicted_gr))

      # k-means for random selected markers
      predicted_random = as.vector(kmeans(log10((raw_data %>% select(all_of(sample(colnames(raw_dataframe), i))))+1), centers=5)$cluster)
      random_scores = append(random_scores, AMI(true_clusters, predicted_random))
    }
    res = rbind(res, data.frame("n_feat"=rep(i,nrep), "AMI"=js_scores, "method"=rep("Jensen-Shannon", nrep)))
    res = rbind(res, data.frame("n_feat"=rep(i,nrep), "AMI"=morans_scores, "method"=rep("Morans I", nrep)))
    res = rbind(res, data.frame("n_feat"=rep(i,nrep), "AMI"=gr_scores, "method"=rep("Global Gene Relevance", nrep)))
    res = rbind(res, data.frame("n_feat"=rep(i,nrep), "AMI"=random_scores, "method"=rep("Random", nrep)))
  }
  return(res)
}

test = evaluateAdjustedMutualInformation(metadata$cluster, js.ranking, morans.ranking, gr.ranking, raw_data, 5)
test %>% group_by(method, n_feat) %>%
  mutate(avg_AMI = mean(AMI)) %>%
  select(-AMI) %>%
  distinct() %>%
  ggplot(aes(x = n_feat, y = avg_AMI)) + geom_point(aes(color = method)) + geom_line(aes(color = method)) + 
  labs(x = "Number of markers", y = "Mean Adjusted Mutual Information (AMI)", title = "Evaluating ability of markers selected to classify clusters")

# ===========================================================================================================================
























evaluateAdjustedMutualInformation <- function(true_clusters, gene_ranking, raw_dataframe, nrep) {
  res = data.frame("n_feat" = NULL, "AMI" = NULL, "condition" = NULL, stringsAsFactors = FALSE)
  for(i in 2:50) {
    print(paste("evaluating for marker number: ", i))
    actual_scores = vector()
    random_scores = vector()
    # calculate UMAP embedding on actual and random selected markers
    actual_umap = umap(log10((raw_data %>% select(gene_ranking[1:i]))+1), init = "spectral", verbose = FALSE)
    random_umap = umap(log10((raw_data %>% select(all_of(sample(colnames(raw_dataframe), i))))+1), init = "spectral", verbose = FALSE)
    for(j in 1:nrep) {
      # k-means for actual selected markers
      predicted_clustering = as.vector(kmeans(actual_umap[,-3], centers=5)$cluster)
      score = AMI(true_clusters, predicted_clustering)
      actual_scores = append(actual_scores, score)
      
      # k-means for random selected markers
      random_clustering = as.vector(kmeans(random_umap[,-3], centers=5)$cluster)
      random_score = AMI(true_clusters, random_clustering)
      random_scores = append(random_scores, random_score)
    }
    res = rbind(res, data.frame("n_feat"=rep(i,nrep), "AMI"=actual_scores, "condition"=rep("actual", nrep)))
    res = rbind(res, data.frame("n_feat"=rep(i,nrep), "AMI"=random_scores, "condition"=rep("random", nrep)))
  }
  return(res)
}


umap <- umap(log10((raw_data %>% select(js.ranking[1:5]))+1), init = "spectral", verbose = FALSE)
vis_umap = cbind(as.data.frame(umap), "Group" = metadata$cluster)


ggplot(vis_umap, aes(x = V1, y = V2)) + geom_point(aes(fill = as.factor(Group)), pch = 21, color = 'black', alpha = 1, size = 2) +
  labs(x = "UMAP1", y = "UMAP2", title = "UMAP on full dataset") + theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.title.x = element_text(size = 18),
        axis.title.y = element_text(size= 18),
        axis.text.x = element_text(size =13),
        axis.text.y = element_text(size = 13))

evaluateSilhouette <- function(true_clusters, gene_ranking, raw_dataframe) {
  res = data.frame("n_feat" = NULL, "sil_score" = NULL, stringsAsFactors = FALSE)
  for(i in 0:length(gene_ranking)) {
    if(i == 0) {
      res = rbind(res, data.frame("n_feat"=0, "sil_score"=0))
    } else {
      data_subset = raw_data %>% select(gene_ranking[1:i])
      dist_matrix = daisy(data_subset, metric = "euclidean")
      sil = silhouette(as.numeric(true_clusters), dist_matrix)
      mean_sil_score = mean(sil[,3])
      res = rbind(res, data.frame("n_feat"=i, "sil_score"=mean_sil_score))
    }
  }
  return(res)
}


umap <- umap(log10(raw_data+1), init = "spectral", verbose = FALSE)
actual_clustering = as.vector(kmeans(umap[,-3], centers=4)$cluster)

umap <- umap(log10((raw_data %>% select(all_of(js.ranking)))+1), init = "spectral", verbose = FALSE)
nearly_clustering = as.vector(kmeans(umap[,-3], centers=4)$cluster)

AMI(actual_clustering, nearly_clustering)



predicted_clustering = cutree(hclust(dist(raw_data)), 3)
predicted_clustering = as.vector(kmeans(raw_data, centers=3)$cluster)

library(uwot)
test = umap(raw_data)


predicted_clustering = as.vector(kmeans(raw_data %>% select(js.ranking[1:20]), centers=3)$cluster)




test = evaluateSilhouette(metadata$cluster, js.ranking, raw_data)



# explainer result
explainer.res = read.csv('../../results/explainer_result_post=2.csv', header = FALSE)
colnames(explainer.res) = colnames(raw_data)
rownames(explainer.res) = unique(metadata$cluster)
explainer.res = explainer.res %>% tibble::rownames_to_column(var="cluster") %>%
  gather(gene_short_name, relevance, -cluster)
explainer.ranking = unique(explainer.res[order(-explainer.res$relevance),]$gene_short_name) # order by relevance



good = colnames(raw_data[,c(1:20)])


test = raw_data %>% select(all_of(good))
diss_matrix = daisy(test, metric = "euclidean")
sil = silhouette(as.numeric(metadata$cluster), diss_matrix)
sil@avg.width
plot(sil)
mean(sil[,3])
