# combine cfos data with PL projection patterns

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.core.reference_space_cache import ReferenceSpaceCache
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import ttest_ind
from statsmodels.stats import multitest
import tifffile as tf
warnings.filterwarnings('ignore')
cfos_path = r"C:\Users\shane\Dropbox (ListonLab)\shane\python_projects\get_atlas_proj_density\data\Cohort6_cfos_6_22_21_combined_results - all_density_excl.csv"
combined_path = r"C:\Users\shane\Dropbox (ListonLab)\shane\python_projects\get_atlas_proj_density\data\cfos_projection_combined.csv"
combined_pl_proj_path = r"C:\Users\shane\Dropbox (ListonLab)\shane\python_projects\get_atlas_proj_density\data\cfos_projection_combined_plProj.csv"
region_list_path = r"\\PC300694.med.cornell.edu\homes\SmartSPIM_Data\2022_01_19\20220119_16_47_57_SJ0612_destriped_DONE\full_brain_regions_LR.csv"
anno25_path = r"\\PC300694.med.cornell.edu\homes\SmartSPIM_Data\2022_01_19\20220119_16_47_57_SJ0612_destriped_DONE\annotation_25_full_transverse_LR.tiff"


def get_atlas_data():
    mcc = MouseConnectivityCache(manifest_file='connectivity/mouse_connectivity_manifest.json')
    reference_space_key = 'annotation/ccf_2017'
    resolution = 25
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
    rsp = rspc.get_reference_space()
    # all_experiments = mcc.get_experiments(dataframe=True)
    structure_tree = mcc.get_structure_tree()
    return mcc, rsp, structure_tree


def get_cfos_data(cfos_path):
    cfos = pd.read_csv(cfos_path)
    return cfos


def get_pl_projections(structure_tree, mcc):
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


def set_structure_names(structure_tree, proj)
    structure_names = pd.DataFrame(structure_tree.nodes(proj.structure_id)).reset_index()
    proj["name"] = structure_names.name
    proj["acronym"] = structure_names.acronym
    hemi_dict = {1: "right ", 2:"left "} # this is flipped relative to the true mappings. This is because our mice have left side injections and the Allen Atlas data is for right side
    name_LR = [hemi_dict[i] for i in proj["hemisphere_id"]] + proj["name"]
    proj["name"] = name_LR
    return proj_named


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
            [row[1].ChR2_SJ0619,
            row[1].ChR2_SJ0602,
            row[1].ChR2_SJ0603,
            row[1].ChR2_SJ0605,
            row[1].ChR2_SJ0612],
            [row[1].YFP_SJ0601,
            row[1].YFP_SJ0604,
            row[1].YFP_SJ0606,
            row[1].YFP_SJ0610,
            row[1].YFP_SJ0613,
            row[1].YFP_SJ0615],
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

check_for_data(combined_path, "P_val_corrected")
#combined = pd.read_csv(combined_path)
# signif_names = combined.name[(combined["p_val_corrected"]<0.05)]


def plot_pvals(combined):
    plt.plot(combined["p_val"])
    plt.plot(combined["p_val_corrected"])
    plt.show()


def restrict_to_pl_proj(combined):
    combined_pl_proj = combined[(combined["projection_density"] > 0)]
    combined_pl_proj = combined_pl_proj[(combined_pl_proj["projection_intensity"] > 0)]
    combined_pl_proj = combined_pl_proj[(combined_pl_proj["projection_energy"] > 0)]
    combined_pl_proj["pd_log"] = np.log(combined_pl_proj["projection_density"]+0.0000001)
    combined_pl_proj["pe_log"] = np.log(combined_pl_proj["projection_energy"]+0.0000001)
    combined_pl_proj["pi_log"] = np.log(combined_pl_proj["projection_intensity"]+0.0000001)
    return combined_pl_proj


# combined_pl_proj_exists = check_for_data(combined_pl_proj_path, "pi_log")
# if not combined_pl_proj_exists:
# combined_pl_proj.to_csv(combined_pl_proj_path)
# structures - lifecanvas

def get_life_canvas_data(region_list_path, anno25_path)
    region_list = pd.read_csv(region_list_path)
    parent_child_dict = region_list.set_index("parent_structure_id")["id"].to_dict()
    child_parent_dict = region_list.set_index("id")["parent_structure_id"].to_dict()
    region_list["grandparent_structure_id"] = region_list.parent_structure_id.map(child_parent_dict).fillna(-1).astype(int)



# region_list["x_loc"] = ""
# region_list["y_loc"] = ""
# region_list["z_loc"] = ""
# anno25 = tf.imread(anno25_path)

# anno25_unique = np.unique(anno25)

# for i, id in enumerate(region_list["id"]):
#     anno_thresh = anno25==id
#     print(np.max(anno_thresh))
#     region_list["x_loc"].iloc[i] = np.mean(np.where(anno_thresh)[0])
#     region_list["y_loc"].iloc[i] = np.mean(np.where(anno_thresh)[1])
#     region_list["z_loc"].iloc[i] = np.mean(np.where(anno_thresh)[2])

structure_unionizes["centroid_x"] = ""
structure_unionizes["centroid_y"] = ""
structure_unionizes["centroid_z"] = ""

for i, id in enumerate(structure_unionizes["structure_id"]):
    print(id)
    mask = rsp.make_structure_mask([id])
    # centroid = [np.mean(x_value) for x_value in np.where(mask)]
    # structure_unionizes["centroid_x"] = centroid[0]
    # structure_unionizes["centroid_y"] = centroid[1]
    # structure_unionizes["centroid_z"] = centroid[2]
