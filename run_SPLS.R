library(spls)
library(readr)


cfos <- read_csv('C:\\Users\\shane\\Dropbox (ListonLab)\\shane\\python_projects\\cfos_regressions\\results\\cfos_kept_regions.csv')

expr <- read_csv('C:\\Users\\shane\\Dropbox (ListonLab)\\shane\\python_projects\\cfos_regressions\\results\\expr_kept_regions.csv')

expr_name <- expr$acronym

expr_mat <- expr[,2:ncol(expr)]


cfos$effect_size_zero_mean <- (cfos$effect_size - mean(cfos$effect_size))

cfos$effect_size_scale <- cfos$effect_size_zero_mean / sd(cfos$effect_size_zero_mean)


# generate random permutation matrix
num_bootstraps <- 10000
num_rows <- dim(expr_mat)[1]
perm_mat_rand <- matrix(0,num_rows, num_bootstraps)
for (i in 1:num_rows){
  perm_mat_rand[,i] = sample.int(num_rows, replace=FALSE)
}


# null model using random permutations
for (i in 1:num_bootstraps){
  mdl <- spls(x=expr_mat, y=cfos$effect_size_scale[perm_mat_rand[,i]], K=2, eta=0.7, select="pls2", fit="simpls", scale.x=FALSE, scale.y=FALSE, eps=1e-4, maxstep=100, trace=FALSE)
}

test <- spls(x=expr_mat, y=cfos$effect_size_scale, K=2, eta=0.7, select="pls2", fit="simpls", scale.x=FALSE, scale.y=FALSE, eps=1e-4, maxstep=100, trace=FALSE)
coef_test <- coef(test)



