# combine cfos data with PL projection patterns

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.core.reference_space_cache import ReferenceSpaceCache
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import ttest_ind
from statsmodels.stats import multitest
import tifffile as tf
from sklearn.cross_decomposition import PLSRegression
warnings.filterwarnings('ignore')
cfos_path = r"C:\Users\shane\Dropbox (ListonLab)\shane\python_projects\get_atlas_proj_density\data\Cohort6_cfos_6_22_21_combined_results - all_density_excl.csv"
combined_path = r"C:\Users\shane\Dropbox (ListonLab)\shane\python_projects\get_atlas_proj_density\data\cfos_projection_combined.csv"
combined_pl_proj_path = r"C:\Users\shane\Dropbox (ListonLab)\shane\python_projects\get_atlas_proj_density\data\cfos_projection_combined_plProj.csv"
region_list_path = r"\\PC300694.med.cornell.edu\homes\SmartSPIM_Data\2022_01_19\20220119_16_47_57_SJ0612_destriped_DONE\full_brain_regions_LR.csv"
anno25_path = r"\\PC300694.med.cornell.edu\homes\SmartSPIM_Data\2022_01_19\20220119_16_47_57_SJ0612_destriped_DONE\annotation_25_full_transverse_LR.tiff"
pl_proj_path = r"C:\Users\shane\Dropbox (ListonLab)\shane\python_projects\get_atlas_proj_density\data\cfos_projection_combined_plProj_withCentroids.csv"
exp_path = r"C:\Users\shane\Dropbox (ListonLab)\shane\python_projects\pls_regression\data\structure_unionizes_all_mouse_expression_density.csv"
exp_rep_path = r"C:\Users\shane\Dropbox (ListonLab)\shane\python_projects\pls_regression\data\structure_unionizes_all_mouse_expression_density_rep.csv"


def get_atlas_data():
    mcc = MouseConnectivityCache(manifest_file='connectivity/mouse_connectivity_manifest.json')
    reference_space_key = 'annotation/ccf_2017'
    resolution = 25
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
    rsp = rspc.get_reference_space()
    # all_experiments = mcc.get_experiments(dataframe=True)
    structure_tree = mcc.get_structure_tree()
    name_map = structure_tree.get_name_map()
    return mcc, rsp, structure_tree, name_map


def get_cfos_data(cfos_path):
    cfos = pd.read_csv(cfos_path)
    return cfos


def get_pl_proj(structure_tree, mcc):
    pl = structure_tree.get_structures_by_acronym(['PL'])[0]
    pl_experiment = mcc.get_experiments(
        cre=False,
        injection_structure_ids=[pl['id']],
        )
    pl_exp_ids = [i['id'] for i in pl_experiment]
    structure_unionizes = mcc.get_structure_unionizes(
        [pl_exp_ids[0]],
        is_injection=False,
        hemisphere_ids=[1, 2],
        include_descendants=True).reset_index(drop=True)
    proj = structure_unionizes.sort_values(
        by=["projection_energy"],
        ascending=True).reset_index(drop=True)
    return proj


def set_structure_names(structure_tree, proj):
    structure_names = pd.DataFrame(structure_tree.nodes(proj.structure_id)).reset_index()
    proj["name"] = structure_names.name
    proj["acronym"] = structure_names.acronym
    hemi_dict = {1: "right ", 2:"left "} # this is flipped relative to the true mappings. This is because our mice have left side injections and the Allen Atlas data is for right side
    name_LR = [hemi_dict[i] for i in proj["hemisphere_id"]] + proj["name"]
    proj["name"] = name_LR
    proj["hemi_LR"] = [hemi_dict[i] for i in proj["hemisphere_id"]]
    return proj


def check_for_data(path, last_column):
    combined_exists = os.path.exists(path)
    if combined_exists:
        combined_head = pd.read_csv(path, index_col=0, nrows=0).columns.tolist()
        if last_column in combined_head:
            return True
        else:
            return False
    else:
        return False


def combine_cfos_and_projections(cfos, proj):
    cfos_not_in_proj = cfos.name[np.logical_not(cfos.name.isin(proj.name))]
    proj_not_in_cfos = proj.name[np.logical_not(proj.name.isin(cfos.name))]
    combined = cfos.merge(proj, how="inner", on = "name")
    print(f"Creating combined dataset.")
    print(f"Lost regions from cfos data are: {cfos_not_in_proj.values}")
    print(f"Lost regions from projection data are: {proj_not_in_cfos.values}")
    return combined


def append_statistics(combined, combined_path):
    t_stats = []
    p_vals = []
    for row in combined.iterrows():
        t_stat, p_val = ttest_ind(
            [
                row[1].ChR2_SJ0619,
                row[1].ChR2_SJ0602,
                row[1].ChR2_SJ0603,
                row[1].ChR2_SJ0605,
                row[1].ChR2_SJ0612
                ],
            [
                row[1].YFP_SJ0601,
                row[1].YFP_SJ0604,
                row[1].YFP_SJ0606,
                row[1].YFP_SJ0610,
                row[1].YFP_SJ0613,
                row[1].YFP_SJ0615
                ],
            axis=0,
            equal_var=True,
            nan_policy='propagate',
            permutations=None,
            random_state=None,
            alternative='two-sided',
            trim=0)
        t_stats.append(t_stat)
        p_vals.append(p_val)
    p_vals = np.nan_to_num(p_vals, nan=1)
    t_stats = np.nan_to_num(t_stats, nan=0)
    combined["t_stat"] = t_stats
    combined["p_val"] = p_vals
    corrected = multitest.multipletests(p_vals, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
    combined["p_val_corrected"] = corrected[1]
    combined.to_csv(combined_path)
    return combined


def plot_pvals(combined):
    plt.plot(combined["p_val"])
    plt.plot(combined["p_val_corrected"])
    plt.show()


def restrict_to_pl_proj(combined):
    combined_pl_proj = combined[(combined["projection_density"] > 0)]
    combined_pl_proj = combined_pl_proj[(combined_pl_proj["projection_intensity"] > 0)]
    combined_pl_proj = combined_pl_proj[(combined_pl_proj["projection_energy"] > 0)]
    return combined_pl_proj


def log_transform(proj):
    proj["pd_log"] = np.log(proj["projection_density"]+0.0000001)
    proj["pe_log"] = np.log(proj["projection_energy"]+0.0000001)
    proj["pi_log"] = np.log(proj["projection_intensity"]+0.0000001)
    return combined


def get_life_canvas_data(region_list_path, anno25_path):
    region_list = pd.read_csv(region_list_path)
    # parent_child_dict = region_list.set_index("parent_structure_id")["id"].to_dict()
    child_parent_dict = region_list.set_index("id")["parent_structure_id"].to_dict()
    region_list["g_parent_structure_id"] = region_list.parent_structure_id.map(child_parent_dict).fillna(-1).astype(int)
    region_list["g_g_parent_structure_id"] = region_list.g_parent_structure_id.map(child_parent_dict).fillna(-1).astype(int)
    region_list["g_g_g_parent_structure_id"] = region_list.g_g_parent_structure_id.map(child_parent_dict).fillna(-1).astype(int)
    anno25 = tf.imread(anno25_path)
    return region_list, anno25


def get_centroids(proj, rsp, name_map):
    proj["centroid_x"] = ""
    proj["centroid_y"] = ""
    proj["centroid_z"] = ""
    print("Finding centroids")
    for row in proj.iterrows():
        i = row[0]
        id = row[1].structure_id
        hemi = row[1].hemi_LR
        mask = rsp.make_structure_mask([id])
        side_left = hemi == "left"
        if side_left:
            mask_left = mask[:, :, 228:456]
            centroid_left = [np.mean(x_value) for x_value in np.where(mask_left)]
            proj["centroid_x"].iloc[i] = centroid_left[0]
            proj["centroid_y"].iloc[i] = centroid_left[1]
            proj["centroid_z"].iloc[i] = centroid_left[2]
        else:
            mask_right = mask[:, :, 0:228]
            centroid_right = [np.mean(x_value) for x_value in np.where(mask_right)]
            proj["centroid_x"].iloc[i] = centroid_right[0]
            proj["centroid_y"].iloc[i] = centroid_right[1]
            proj["centroid_z"].iloc[i] = centroid_right[2]
        print(f"{np.round(((i+1)/proj.shape[0])*100,2)}% done")
    return proj


# function to get expression data?
def get_expression_data():
    exp = pd.read_csv(exp_path)
    exp = exp.groupby(["gene_id", "acronym"]).agg({"expression_density": "mean"}).reset_index()
    exp["ed_dm_scale"] = (exp["expression_density"] - np.mean(exp["expression_density"]))/np.std(exp["expression_density"])
    exp = exp.pivot(index="acronym", columns="gene_id", values="ed_dm_scale")
    exp_idx = exp.index
    exp_idx_rep = np.repeat(exp_idx, 2) + np.tile(["-L","-R"], exp.shape[0])
    exp_col = exp.columns
    exp_rep = pd.DataFrame(np.repeat(exp.values,2,axis=0))
    exp_rep.index = exp_idx_rep
    exp_rep.columns = exp_col
    return exp


def get_dist_to_pl(pl_proj):
    pl_x = pl_proj["centroid_x"][pl_proj["name"] == 'left Prelimbic area'].values[0]
    pl_y = pl_proj["centroid_y"][pl_proj["name"] == 'left Prelimbic area'].values[0]
    pl_z = pl_proj["centroid_z"][pl_proj["name"] == 'left Prelimbic area'].values[0]
    pl_proj["dist_to_pl"] = ""
    for row in pl_proj.iterrows():
        i = row[0]
        pl_proj["dist_to_pl"].iloc[i] = np.sqrt(
                (
                    (row[1].centroid_x - pl_x)**2 + (row[1].centroid_y - pl_y)**2 + (row[1].centroid_z - pl_z)**2
                    )
        )
    return pl_proj


def match_cfos_to_exp(pl_proj, exp):
    pl_proj = pl_proj.set_index("acronym_x") # note: should merge on name and acronym so that acronym isn't duplicated
    if (exp_rep.index.name != "acronym"):
        print("Error: expression data is missing acronym index")
        return ""
    else:
        pl_proj_sub = pl_proj[pl_proj.index.isin(exp.index)]
        exp_sub = exp[exp.index.isin(pl_proj_sub.index)]
        pl_proj_lost_regions = pl_proj.index[np.logical_not(pl_proj.index.isin(exp.index))]
        exp_lost_regions = exp.index[np.logical_not(exp.index.isin(pl_proj.index))]
        pl_proj_sub_sort = pl_proj_sub.sort_index()
        exp_sub_sort = exp_sub.sort_index()
        print(f"Regions in projection/cfos data missing from expression data: {pl_proj_lost_regions}")
        print(f"Regions in expression data missing from projection/cfos data: {exp_lost_regions}")
        return pl_proj_sub_sort, exp_sub_sort, pl_proj_lost_regions, exp_lost_regions

# Plan: generate random permutation matrix.
# Alex creates a range of values to permute over.
# The values range over the number of data rows in the expression matrix
bootstrap_count = 10000
temp_range = exp.shape[0]
perm_mat_rand = np.zeros((exp.shape[0],bootstrap_count))
for i in np.arange(bootstrap_count):
    perm_mat_rand[:,i] = np.random.permutation(temp_range)





# null model using random permutation of rows
for i in np.arange(bootstrap_count):
    idx= perm_mat_rand[:,i]
    X = 
    pls_mdl = PLSRegression(n_components=1, scale=False)
    pls_mdl.fit()




def main():
    mcc, rsp, structure_tree, name_map = get_atlas_data()
    cfos = get_cfos_data(cfos_path)
    proj = get_pl_proj(structure_tree, mcc)
    proj = set_structure_names(structure_tree, proj)
    if check_for_data(combined_path, "p_val_corrected"):
        combined = pd.read_csv(combined_path)
    else:
        combined = combine_cfos_and_projections(cfos, proj)
        combined = append_statistics(combined, combined_path)
    pl_proj = restrict_to_pl_proj(combined)
    pl_proj = log_transform(pl_proj)
    # region_list, anno25 = get_life_canvas_data(region_list_path, anno25_path)
    pl_proj = get_centroids(pl_proj, rsp, name_map)
    pl_proj = get_dist_to_pl(pl_proj)
    pl_proj.to_csv(pl_proj_path)
    pl_proj_sub, exp_sub, cfos_lost_regions, expr_lost_regions = match_cfos_to_exp(pl_proj, exp) # why are we missing so many regions (~400)
    
