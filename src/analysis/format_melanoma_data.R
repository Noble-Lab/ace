setwd('~/Documents/Github/2020_ylu465_group_explain/src/analysis')

data = read.table('../../data/GSE72056_melanoma_single_cell_revised_v2.txt', header = FALSE)
data = data[-23206, ] # duplicated gene, MARCH2. Also in 17012 row
data = data[-c(10194, 16350, 23640), ] # duplicated gene, MARCH1
formatted_data = data %>% remove_rownames %>% tibble::column_to_rownames(var = "V1") %>% t() %>% as.data.frame()
non_malignant = formatted_data %>% filter(`malignant(1=no,2=yes,0=unresolved)` == 1) %>%
  filter(`non-malignant cell type (1=T,2=B,3=Macro.4=Endo.,5=CAF;6=NK)` != 0)

row_names = vector()
for(i in 1:nrow(non_malignant)) {
  row_names = append(row_names, paste(i,"_",non_malignant[,4][i], sep=""))
}

rownames(non_malignant) = row_names
metadata = non_malignant[,c(1:4)]
final_data = non_malignant[,-c(1:4)]

write.csv(final_data, '../../data/non_malignant_single_cell.csv', row.names = TRUE)
write.csv(metadata, '../../data/non_malignant_meta.csv', row.names = TRUE)

final_data = read.table('../../data/non_malignant_single_cell.csv', header = TRUE, row.names = 1, sep = ',')
metadata = read.table('../../data/non_malignant_meta.csv', header = TRUE, row.names = 1, sep = ',')



library(uwot)

set.seed(123)
umap <- umap((log(final_data)+1), init = "spectral", verbose = FALSE, pca = 50)
vis_umap = cbind(as.data.frame(umap), "Group" = as.factor(metadata$non.malignant.cell.type..1.T.2.B.3.Macro.4.Endo..5.CAF.6.NK.))

