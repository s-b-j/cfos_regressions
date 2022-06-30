cfos_table_in = readtable("C:\Users\shane\Dropbox (ListonLab)\shane\python_projects\cfos_regressions\results\cfos_kept_genes.csv");

rsFC_vector = cfos_table_in.effect_size;

expr_mat_in = readmatrix("C:\Users\shane\Dropbox (ListonLab)\shane\python_projects\cfos_regressions\results\expr_kept_genes.csv");

region_names = cfos_table_in.acronym_x; 
gene_ids = expr_mat_in(1,:);

ROI_expression_mat = expr_mat_in(2:end,2:end);

ROI_expression_mat(2:end,:) = (ROI_expression_mat(2:end,:) - mean(ROI_expression_mat(2:end,:))) ./ std(ROI_expression_mat(2:end,:));
rsFC_vector = (rsFC_vector - mean(rsFC_vector)) ./ std(rsFC_vector);



%% Generate random permutation matrix

%set bootstrap count
bootstrap_count = 10000;

%set random seed
rng(1)
%generate random permutation matrix using 'randperm'
temp_range = size(ROI_expression_mat,1);
for i = 1:bootstrap_count
    perm_mat_rand(:,i) = randperm(temp_range);
end
    
%% Run PLS to determine model loading score correlation and p-values

warning off %suppress non-sparsity warning in 'spls' function

%null model using random permutation of Y-vector rows
for i = 1:bootstrap_count
    [gene_loading_weights,temp_v] = spls(ROI_expression_mat, rsFC_vector(perm_mat_rand(:,i)), 1000,1);
    null_corr_rand(i) = corr(ROI_expression_mat * gene_loading_weights, rsFC_vector(perm_mat_rand(:,i)) * temp_v);
end


%compare empirical correlation to null distribution
[gene_loading_weights,~] = spls(ROI_expression_mat,rsFC_vector,1000,1);
model_corr = corr(ROI_expression_mat * gene_loading_weights,rsFC_vector);
p_rand = sum(null_corr_rand(1:bootstrap_count) > model_corr)/bootstrap_count;


%% rank genes by PLS loading weight

%set random seed
rng(1)

%generate null distribution of gene loading weights using
%bootstrap-resampling procedure
null_temp_u = zeros(bootstrap_count,size(ROI_expression_mat,2));
for i = 1:bootstrap_count
    [null_ROI_expression_mat,null_idx] = datasample(ROI_expression_mat,size(ROI_expression_mat,1));
    resampled_rsFC_vector = rsFC_vector(null_idx);
    [null_temp_u(i,:),~] = spls(null_ROI_expression_mat,resampled_rsFC_vector,1000,1);
end


%correct gene_loading_weights according to stability
gene_loading_weights = gene_loading_weights ./ std(null_temp_u,0,1)';
 
%generate gene ranklist 
[~,temp_gene_ranks] = sort(gene_loading_weights,'descend');
gene_LW_ranklist = ROI_expression_mat(1,temp_gene_ranks)';
gene_LW_ranklist = gene_ids(temp_gene_ranks)';
gene_LW_ranklist(:,2) = gene_loading_weights(temp_gene_ranks);

gene_LW_ranklist = rmmissing(gene_LW_ranklist);
gene_table = readtable("C:\Users\shane\Dropbox (ListonLab)\shane\python_projects\pls_regression\data\structure_unionizes_all_mouse_expression.csv");

gene_table = gene_table(:,2:3);

gene_table = rmmissing(gene_table);
gene_table = unique(gene_table,'rows');

gene_map = containers.Map(gene_table.gene_id, gene_table.gene_symbol);

gene_ids_ranked = gene_LW_ranklist(:,1);

final_idx = size(gene_ids_ranked,1);

list_out = [""];

for i = 1:final_idx
    list_out = [list_out; gene_map(gene_ids_ranked(i))];
end

