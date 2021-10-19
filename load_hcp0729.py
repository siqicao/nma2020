


############### Contributor ###############
# Yuang Chen @ Peking University
# Siqi Cao @ Chineses Academy of Sciences, Institute of Psychology
# please inform contributors if you are interested in this script!
# Email: caosq@psych.ac.cn


import os
from typing import Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from numpy.core._multiarray_umath import ndarray
from scipy.stats import norm
import statsmodels.stats.multitest
from nilearn import plotting, datasets
import seaborn as sns
import pandas as pd

# basic parameters
HCP_DIR = "./hcp"
# The data shared for NMA projects is a subset of the full HCP dataset
N_SUBJECTS = 339
# The data have already been aggregated into ROIs from the Glasesr parcellation
N_PARCELS = 360
# The acquisition parameters for all tasks were identical
TR = 0.72  # Time resolution, in sec
# The parcels are matched across hemispheres with the same order
HEMIS = ["Right", "Left"]
# Each experiment was repeated multiple times in each subject
N_RUNS_REST = 4
N_RUNS_TASK = 2
N_FEAR_condition = 100
N_NEUT_condition = 150
N_FEAR_trials = 24
N_NEUT_trials = 36
N_FEAR_blocks = 4
N_NEUT_blocks = 6
N_TRIAL = 4
N_BLOCK = 25
# Time series data are organized by experiment, with each experiment
# having an LR and RL (phase-encode direction) acquistion
BOLD_NAMES = [
  "rfMRI_REST1_LR", "rfMRI_REST1_RL",
  "rfMRI_REST2_LR", "rfMRI_REST2_RL",
  "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR",
  "tfMRI_WM_RL", "tfMRI_WM_LR",
  "tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR",
  "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR",
  "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR",
  "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR",
  "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"
]
subjects = range(N_SUBJECTS)
regions = np.load(f"{HCP_DIR}/regions.npy").T
region_info = dict(
    name=regions[0].tolist(),
    network=regions[1],
    myelin=regions[2].astype(np.float),
)

# data loading functions
def get_image_ids(name):
    """Get the 1-based image indices for runs in a given experiment.

      Args:
        name (str) : Name of experiment ("rest" or name of task) to load
      Returns:
        run_ids (list of int) : Numeric ID for experiment image files

    """
    run_ids = [
        i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code
    ]
    if not run_ids:
        raise ValueError(f"Found no data for '{name}''")
    return run_ids
def load_timeseries(subject, name, runs=None, concat=True, remove_mean=True):
    """Load timeseries data for a single subject.

    Args:
      subject (int): 0-based subject ID to load
      name (str) : Name of experiment ("rest" or name of task) to load
      run (None or int or list of ints): 0-based run(s) of the task to load,
        or None to load all runs.
      concat (bool) : If True, concatenate multiple runs in time
      remove_mean (bool) : If True, subtract the parcel-wise mean

    Returns
      ts (n_parcel x n_tp array): Array of BOLD data values

    """
    # Get the list relative 0-based index of runs to use
    if runs is None:
        runs = range(N_RUNS_REST) if name == "rest" else range(N_RUNS_TASK)
    elif isinstance(runs, int):
        runs = [runs]

    # Get the first (1-based) run id for this experiment
    offset = get_image_ids(name)[0]

    # Load each run's data
    bold_data = [
        load_single_timeseries(subject, offset + run, remove_mean) for run in runs
    ]

    # Optionally concatenate in time
    if concat:
        bold_data = np.concatenate(bold_data, axis=-1)

    return bold_data
def load_single_timeseries(subject, bold_run, remove_mean=True):
    """Load timeseries data for a single subject and single run.

    Args:
      subject (int): 0-based subject ID to load
      bold_run (int): 1-based run index, across all tasks
      remove_mean (bool): If True, subtract the parcel-wise mean

    Returns
      ts (n_parcel x n_timepoint array): Array of BOLD data values

    """
    bold_path = f"{HCP_DIR}/subjects/{subject}/timeseries"
    bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
    ts = np.load(f"{bold_path}/{bold_file}")
    if remove_mean:
        ts -= ts.mean(axis=1, keepdims=True)
    return ts
def load_evs(subject, name, condition):
    """Load EV (explanatory variable) data for one task condition.

    Args:
      subject (int): 0-based subject ID to load
      name (str) : Name of task
      condition (str) : Name of condition

    Returns
      evs (list of dicts): A dictionary with the onset, duration, and amplitude
        of the condition for each run.

    """
    evs = []
    for id in get_image_ids(name):
        task_key = BOLD_NAMES[id - 1]
        ev_file = f"{HCP_DIR}/subjects/{subject}/EVs/{task_key}/{condition}.txt"
        ev = dict(zip(["onset", "duration", "amplitude"], np.genfromtxt(ev_file).T))
        evs.append(ev)
    return evs


# FMRI slicing functions
def condition_frames(run_evs, skip=0):
    """Identify timepoints corresponding to a given condition in each run.

    Args:
      run_evs (list of dicts) : Onset and duration of the event, per run
      skip (int) : Ignore this many frames at the start of each trial, to account
        for hemodynamic lag

    Returns:
      frames_list (list of 1D arrays): Flat arrays of frame indices, per run

    """
    frames_list = []
    i = 0
    for ev in run_evs:
        frames = []
        # Determine when trial starts, rounded down
        start = np.floor(ev["onset"] / TR).astype(int)

        # Use trial duration to determine how many frames to include for trial
        duration = np.ceil(ev["duration"] / TR).astype(int)

        # Take the range of frames that correspond to this specific trial

        for s, d in zip(start, duration):
            frames = frames + [s + np.arange(skip, d)]

        frames_list.append(np.concatenate(frames))

    return frames_list
def condition_split(timeseries_data, ev, skip=0):
    # Identify the indices of relevant frames
    frames = condition_frames(ev)
    # Select the frames from each image
    selected_data = []
    for run_data, run_frames in zip(timeseries_data, frames):
        selected_data.append(run_data[:, run_frames])

    return selected_data
def block_frames(run_evs, skip=0):
    frames_list = []
    for ev in run_evs:
        frames = []
        # Determine when trial starts, rounded down
        start = np.floor(ev["onset"] / TR).astype(int)
        # Use trial duration to determine how many frames to include for trial
        duration = np.ceil(ev["duration"] / TR).astype(int)
        # Take the range of frames that correspond to this specific trial
        for s, d in zip(start, duration):
            frames = frames + [s + np.arange(skip, d)]
        frames_list.append(frames)

    return frames_list

def block_split(timeseries_data, ev, skip=0):
    # Identify the indices of relevant frames
    frames = block_frames(ev)
    # Select the frames from each image
    selected_data = []
    for run_data, run_frames in zip(timeseries_data, frames):
        selected_data.append(run_data[:, run_frames])

    return selected_data

def trial_frames(run_evs, skip=0):
    frames_list = []
    for ev in run_evs:
        frames = []
        # Determine when trial starts, rounded down
        temponset=ev["onset"]
        trueonset=[]
        for i in temponset:
            for j in np.arange(0,18,3):
                trueonset.append(i+j)
        trueonset=np.array(trueonset)
        start = np.floor(trueonset / TR).astype(int)

        # Use trial duration to determine how many frames to include for trial
        tempduration=ev["duration"]
        trueduration=np.ones_like(start)*3
        duration = np.floor(trueduration/ TR).astype(int)

        # Take the range of frames that correspond to this specific trial

        for s, d in zip(start, duration):
            frames = frames + [s + np.arange(skip, d)]

        frames_list.append(frames)

    return frames_list

def trial_split(timeseries_data,ev,skip=0):
    # Identify the indices of relevant frames
    frames = trial_frames(ev)
    # Select the frames from each image
    selected_data = []
    for run_data, run_frames in zip(timeseries_data, frames):
        selected_data.append(run_data[:, run_frames])

    return selected_data


# if concat = False
# the first bracket refers to the ith subject
# the second bracket refers to the jth run
# the third bracket refers to the kth brain region
# the fourth bracket refers to the tth time points
# if concat == True
# the second brackets omits

if not os.path.isfile("timeseries_task.npy"):
    timeseries_task = []
    for subject in subjects:
      timeseries_task.append(load_timeseries(subject, "emotion", concat=False))
    np.save("timeseries_task.npy",timeseries_task)
else:
    timeseries_task=np.load("timeseries_task.npy")
if not os.path.isfile("fear_trials.npy") or not os.path.isfile("neut_trials.npy"):
    task = "emotion"
    conditions = ["fear", "neut"]  # Run a substraction analysis between two conditions.
    afterpreprocessing=[]
    fear_trials=np.zeros((N_SUBJECTS,N_PARCELS,N_FEAR_trials,N_TRIAL))
    neut_trials=np.zeros((N_SUBJECTS,N_PARCELS,N_NEUT_trials,N_TRIAL))
    for subject in subjects:
      evs = [load_evs(subject, task, cond) for cond in conditions]
      afterpreprocessing.append([trial_split(timeseries_task[subject], ev) for ev in evs])
      temp=np.hstack((afterpreprocessing[subject][0][0],afterpreprocessing[subject][0][1]))
      fear_trials[subject]=temp
      temp=np.hstack((afterpreprocessing[subject][1][0],afterpreprocessing[subject][1][1]))
      neut_trials[subject]=temp
    scipy.io.savemat('trial_data.mat', mdict={'fear_trials': fear_trials,'neut_trials':neut_trials})
    np.save("fear_trials.npy",fear_trials)
    np.save("neut_trials.npy",neut_trials)
else:
    fear_trials=np.load("fear_trials.npy")
    neut_trials=np.load("neut_trials.npy")
if not os.path.isfile("fear_condition.npy") or not os.path.isfile("neut_condition.npy"):
    task = "emotion"
    conditions = ["fear", "neut"]  # Run a substraction analysis between two conditions
    fear_condition=np.zeros((N_SUBJECTS,N_PARCELS,N_FEAR_condition))
    neut_condition=np.zeros((N_SUBJECTS,N_PARCELS,N_NEUT_condition))
    afterpreprocessing=[]
    for subject in subjects:
      evs = [load_evs(subject, task, cond) for cond in conditions]
      afterpreprocessing.append([condition_split(timeseries_task[subject], ev) for ev in evs])
      temp = np.hstack((afterpreprocessing[subject][0][0], afterpreprocessing[subject][0][1]))
      fear_condition[subject] = temp
      temp = np.hstack((afterpreprocessing[subject][1][0], afterpreprocessing[subject][1][1]))
      neut_condition[subject] = temp
    scipy.io.savemat('conditions_data.mat', mdict={'fear_condition': fear_condition,'neut_condition':neut_condition})
    np.save("fear_condition.npy",fear_condition)
    np.save("neut_condition.npy",neut_condition)
else:
    fear_condition=np.load("fear_condition.npy")
    neut_condition=np.load("neut_condition.npy")
if not os.path.isfile("fear_blocks.npy") or not os.path.isfile("neut_blocks.npy"):
    task = "emotion"
    conditions = ["fear", "neut"]  # Run a substraction analysis between two conditions.
    afterpreprocessing=[]
    fear_blocks=np.zeros((N_SUBJECTS,N_PARCELS,N_FEAR_blocks,N_BLOCK))
    neut_blocks=np.zeros((N_SUBJECTS,N_PARCELS,N_NEUT_blocks,N_BLOCK))
    for subject in subjects:
      evs = [load_evs(subject, task, cond) for cond in conditions]
      afterpreprocessing.append([block_split(timeseries_task[subject], ev) for ev in evs])
      temp=np.hstack((afterpreprocessing[subject][0][0],afterpreprocessing[subject][0][1]))
      fear_blocks[subject]=temp
      temp=np.hstack((afterpreprocessing[subject][1][0],afterpreprocessing[subject][1][1]))
      neut_blocks[subject]=temp
    scipy.io.savemat('block_data.mat', mdict={'fear_blocks': fear_blocks,'neut_blocks':neut_blocks})
    np.save("fear_blocks.npy",fear_blocks)
    np.save("neut_blocks.npy",neut_blocks)
else:
    fear_blocks=np.load("fear_blocks.npy")
    neut_blocks=np.load("neut_blocks.npy")

# regress the stimulus onset activity for each condition
if not os.path.isfile("fear_condition_regressed.npy") or not os.path.isfile("neut_condition_regressed.npy")\
        or not os.path.isfile("fear_beta.npy") or not os.path.isfile("neut_beta.npy"):
    fear_condition_regressed = np.zeros_like(fear_condition)
    fear_beta=np.zeros((N_SUBJECTS,N_PARCELS,2))
    length = fear_condition[0,0].shape[0]
    X = np.ones((length,2))
    thread=np.arange(0,length, 1)*0.72 % 3
    idx = thread > 2
    X[idx,1] = 0
    for subject in subjects:
        for brainregion in range(N_PARCELS):
            Y=fear_condition[subject,brainregion]
            fear_beta[subject,brainregion]=np.linalg.inv(X.T@X)@X.T@Y
            fear_condition_regressed[subject, brainregion]=(Y-X@fear_beta[subject,brainregion])

    neut_condition_regressed = np.zeros_like(neut_condition)
    neut_beta=np.zeros((N_SUBJECTS,N_PARCELS,2))
    length=neut_condition[0,0].shape[0]
    X = np.ones((length,2))
    thread = np.arange(0,length, 1)*0.72 % 3
    idx = thread > 2
    X[idx,1] = 0
    for subject in subjects:
        for brainregion in range(N_PARCELS):
            Y=neut_condition[subject,brainregion]
            neut_beta[subject,brainregion]=np.linalg.inv(X.T@X)@X.T@Y
            neut_condition_regressed[subject, brainregion]=(Y-X@neut_beta[subject,brainregion])
    scipy.io.savemat('conditions_regressed_data.mat',
                     mdict={'fear_condition_regressed': fear_condition_regressed,
                     'neut_condition_regressed':neut_condition_regressed})
    np.save("fear_condition_regressed.npy",fear_condition_regressed)
    np.save("neut_condition_regressed.npy",neut_condition_regressed)
    np.save("fear_beta.npy",fear_beta)
    np.save("neut_beta.npy",neut_beta)
else:
    fear_condition_regressed=np.load("fear_condition_regressed.npy")
    neut_condition_regressed=np.load("neut_condition_regressed.npy")
    fear_beta = np.load("fear_beta.npy")
    neut_beta = np.load("neut_beta.npy")


# get each trial after regression
if not os.path.isfile("fear_trials_regressed.npy") or not os.path.isfile("neut_trials_regressed.npy"):
    fear_trials_regressed=np.zeros_like(fear_trials)
    X=np.array([[1,1],[1,1],[1,1],[1,0]])
    length=fear_trials.shape[2]
    for subject in subjects:
        for brainregion in range(N_PARCELS):
            for i in range(length):
                fear_trials_regressed[subject,brainregion,i,:]=fear_trials[subject,brainregion,i]-X@fear_beta[subject,brainregion]
    neut_trials_regressed=np.zeros_like(neut_trials)
    X=np.array([[1,1],[1,1],[1,1],[1,0]])
    length=neut_trials.shape[2]
    for subject in subjects:
        for brainregion in range(N_PARCELS):
            for i in range(length):
                neut_trials_regressed[subject,brainregion,i,:]=neut_trials[subject,brainregion,i]-X@neut_beta[subject,brainregion]
    scipy.io.savemat('trials_regressed_data.mat', mdict={'fear_trials_regressed': fear_trials_regressed,
                                                         'neut_trials_regressed':neut_trials_regressed})
    np.save("fear_trials_regressed.npy",fear_trials_regressed)
    np.save("neut_trials_regressed.npy",neut_trials_regressed)
else:
    fear_trials_regressed = np.load("fear_trials_regressed.npy")
    neut_trials_regressed = np.load("neut_trials_regressed.npy")


# get each block after regression
if not os.path.isfile("fear_blocks_regressed.npy") or not os.path.isfile("neut_blocks_regressed.npy"):
    fear_blocks_regressed=np.zeros_like(fear_blocks)
    X=np.array([[1,1],[1,1],[1,1],[1,0],[1,0],[1,1],[1,1],[1,0],[1,0],[1,1],[1,1],[1,1],[1,0],
                [1,1],[1,1],[1,1],[1,0],[1,1],[1,1],[1,1],[1,0],[1,1],[1,1],[1,1],[1,0]])
    length=fear_blocks.shape[2]
    for subject in subjects:
        for brainregion in range(N_PARCELS):
            for i in range(length):
                fear_blocks_regressed[subject,brainregion,i,:]=fear_blocks[subject,brainregion,i]-X@fear_beta[subject,brainregion]
    neut_blocks_regressed=np.zeros_like(neut_blocks)
    X=np.array([[1,1],[1,1],[1,1],[1,0],[1,0],[1,1],[1,1],[1,0],[1,0],[1,1],[1,1],[1,1],[1,0],
                [1,1],[1,1],[1,1],[1,0],[1,1],[1,1],[1,1],[1,0],[1,1],[1,1],[1,1],[1,0]])
    length=neut_blocks.shape[2]
    for subject in subjects:
        for brainregion in range(N_PARCELS):
            for i in range(length):
                neut_blocks_regressed[subject,brainregion,i,:]=neut_blocks[subject,brainregion,i]-X@neut_beta[subject,brainregion]
    scipy.io.savemat('blocks_regressed_data.mat', mdict={'fear_blocks_regressed': fear_blocks_regressed,
                                                         'neut_blocks_regressed':neut_blocks_regressed})
    np.save("fear_blocks_regressed.npy",fear_blocks_regressed)
    np.save("neut_blocks_regressed.npy",neut_blocks_regressed)
else:
    fear_blocks_regressed = np.load("fear_blocks_regressed.npy")
    neut_blocks_regressed = np.load("neut_blocks_regressed.npy")


# background connectivity
neut_res_fc = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fear_res_fc = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fear_res_fc_z = np.zeros_like(fear_res_fc)
neut_res_fc_z = np.zeros_like(neut_res_fc)
mean_diff_map = np.ones_like((N_PARCELS,N_PARCELS))
z_map = np.ones_like(mean_diff_map)
p_values = np.ones_like(mean_diff_map)

for sub, ts in enumerate(fear_condition_regressed):
    fear_res_fc[sub] = np.corrcoef(ts)
    fear_res_fc_z[sub]=np.arctanh(fear_res_fc[sub])
for sub, ts in enumerate(neut_condition_regressed):
    neut_res_fc[sub] = np.corrcoef(ts)
    neut_res_fc_z[sub] = np.arctanh(neut_res_fc[sub])

with np.load(f"{HCP_DIR}/atlas.npz") as dobj:
  atlas = dict(**dobj)

fear_res_fc_group = np.mean(fear_res_fc,axis=0)
neut_res_fc_group = np.mean(neut_res_fc,axis=0)
#
# # # correlation map plot
# plt.imshow(fear_res_fc_group, interpolation="none", cmap='PRGn', vmin=-1, vmax=1)
# plt.colorbar()
# plt.title('Background Connectivity(Face)')
# plt.savefig("fear_res_fc.png", dpi =300)
# plt.show()
# #
# plt.imshow(neut_res_fc_group, interpolation="none", cmap='PRGn', vmin=-1, vmax=1)
# plt.colorbar()
# plt.title('Background Connectivity(Shape)')
# plt.savefig("neut_res_fc.png", dpi =300)
# plt.show()

# RT correlation
RT_data = pd.read_csv ('/Volumes/caosq/Study/NMA_project/cya_code/hcp/behavior/emotion.csv')

face_mean = np.ones((339,1))
for i in range(339):
    temp = 0
    for j in range(678):
        if RT_data.iloc[j,0] == i:
            temp += RT_data.iloc[j,4]
    face_mean[i] = temp*0.5

shape_mean = np.ones((339,1))
for i in range(339):
    temp = 0
    for j in range(678,1356):
        if RT_data.iloc[j,0] == i:
            temp += RT_data.iloc[j,4]
    shape_mean[i] = temp*0.5

total_rt = np.vstack((face_mean,shape_mean))

idx_slow = []
idx_fast = []
for i in range(678):
    if total_rt[i] < np.median(total_rt):
        idx_fast.append(i)
    else:
        idx_slow.append(i)

# 拼合两个condition的fc
RT_res_fc_total = np.vstack((fear_res_fc,neut_res_fc))
RT_res_fast = np.ones((len(idx_fast),N_PARCELS,N_PARCELS))
RT_res_slow = np.ones((len(idx_slow),N_PARCELS,N_PARCELS))
# 按照反应时中位数划分快慢index
RT_res_fast = RT_res_fc_total[idx_fast]
RT_res_slow = RT_res_fc_total[idx_slow]

RT_res_fc_total_z = np.vstack((fear_res_fc_z,neut_res_fc_z))
RT_res_fast_z = RT_res_fc_total_z[idx_fast]
RT_res_slow_z = RT_res_fc_total_z[idx_slow]

# fig = plt.gcf()
# mean_RT_diff_map = np.mean(RT_res_fast_z,axis=0)-np.mean(RT_res_slow_z,axis=0)

# RT_cc_map = sns.heatmap(mean_RT_diff_map, cmap='Blues', annot=False)
# RT_cc_map.set_title('Background conncetivity difference(fast vs. slow)')
# # fig.set_size_inches(25, 20)
# fig.savefig('RT_BC',dpi = 300)

# plt.figure()
# plt.imshow(mean_RT_diff_map, interpolation="none", cmap='GnBu')
# plt.colorbar()
# plt.title('Background conncetivity difference(fast vs. slow)')
# plt.savefig("RT_BC_2.png", dpi =300)
# plt.show()
#
# RT_res_fast_group = np.mean(RT_res_fast,axis=0)
# RT_res_slow_group = np.mean(RT_res_fast,axis=0)
#
# # # correlation map plot
# plt.figure()
# plt.imshow(RT_res_fast_group, interpolation="none", cmap='PRGn', vmin=-1, vmax=1)
# plt.colorbar()
# plt.title('Background Connectivity(fast)')
# plt.savefig("fast_res_fc.png", dpi =300)
# plt.show()
#
# plt.figure()
# plt.imshow(RT_res_slow_group, interpolation="none", cmap='PRGn', vmin=-1, vmax=1)
# plt.colorbar()
# plt.title('Background Connectivity(slow)')
# plt.savefig("slow_res_fc.png", dpi =300)
# plt.show()

# background connectivity plot
# fear_plot = plotting.view_connectome(fear_res_fc_group, atlas["coords"], edge_threshold="99%")
# fear_plot.open_in_browser()
# neut_plot = plotting.view_connectome(neut_res_fc_group, atlas["coords"], edge_threshold="99%")
# neut_plot.open_in_browser()


# mean_diff_map = np.mean(fear_res_fc_z,axis=0)-np.mean(neut_res_fc_z,axis=0)
# #create new rvs
# z_map = -abs((mean_diff_map-0)/np.sqrt(1/N_SUBJECTS*(1/(N_FEAR_condition-3)+1/(N_NEUT_condition-3))))
# # z-test
# p_values = scipy.stats.norm.cdf(z_map,0,1)*2
# #FDR correction
# p_FDR_corrected=statsmodels.stats.multitest.fdrcorrection(p_values)
# # FWE correction
# p_mask = p_values <0.05/(360*360)
# p_cond = np.where(p_mask,p_values,-1)
#
# plt.imshow(p_cond, interpolation="none", cmap='PRGn', vmin=-1, vmax=1)
# plt.colorbar()
# plt.title('P map')
# plt.savefig("P map.png", dpi =300)
# plt.show()


# network regress
network_names = np.unique(region_info["network"])
whole_networks = region_info["network"]
idx = []
networks_fear = np.zeros((len(network_names),fear_condition.shape[2]))
networks_neut = np.zeros((len(network_names),neut_condition.shape[2]))
fear_network = np.zeros((N_SUBJECTS, len(network_names), fear_condition.shape[2]))
neut_network = np.zeros((N_SUBJECTS, len(network_names), neut_condition.shape[2]))

for subject in range(N_SUBJECTS):
    i = -1
    for network in network_names:
        i += 1
        idx = np.array(np.where(whole_networks == network))
        temp_fear = fear_condition[subject][idx][0].mean(axis=0)
        temp_neut = neut_condition[subject][idx][0].mean(axis=0)
        networks_fear[i] = temp_fear
        networks_neut[i] = temp_neut
    fear_network[subject]= networks_fear
    neut_network[subject] = networks_neut

# fear condition network regress
fear_network_regressed = np.zeros_like(fear_network)
fear_net_beta=np.zeros((N_SUBJECTS,N_PARCELS,2))
length = fear_network[0,0].shape[0]
X = np.ones((length,2))
thread=np.arange(0,length, 1)*0.72 % 3
idx = thread > 2
X[idx,1] = 0

for subject in subjects:
    for network in range(len(network_names)):
        Y=fear_network[subject,network]
        fear_net_beta[subject,network]=np.linalg.inv(X.T@X)@X.T@Y
        fear_network_regressed[subject, network]=(Y-X@fear_net_beta[subject,network])

# neut condition network regress
neut_network_regressed = np.zeros_like(neut_network)
neut_net_beta=np.zeros((N_SUBJECTS,N_PARCELS,2))
length = neut_network[0,0].shape[0]
X = np.ones((length,2))
thread=np.arange(0,length, 1)*0.72 % 3
idx = thread > 2
X[idx,1] = 0

for subject in subjects:
    for network in range(len(network_names)):
        Y=neut_network[subject,network]
        neut_net_beta[subject,network]=np.linalg.inv(X.T@X)@X.T@Y
        neut_network_regressed[subject, network]=(Y-X@neut_net_beta[subject,network])

# background connectivity
neut_net_res_fc = np.zeros((N_SUBJECTS, len(network_names), len(network_names)))
fear_net_res_fc = np.zeros((N_SUBJECTS, len(network_names), len(network_names)))
fear_net_res_fc_z = np.zeros_like(fear_net_res_fc)
neut_net_res_fc_z = np.zeros_like(neut_net_res_fc)
mean_net_diff_map = np.ones_like((len(network_names),len(network_names)))
z_net_map = np.ones_like(mean_net_diff_map)
p_net_values = np.ones_like(mean_net_diff_map)


for sub, ts in enumerate(fear_network_regressed):
    fear_net_res_fc[sub] = np.corrcoef(ts)
    fear_net_res_fc_z[sub]=np.arctanh(fear_net_res_fc[sub])
for sub, ts in enumerate(neut_network_regressed):
    neut_net_res_fc[sub] = np.corrcoef(ts)
    neut_net_res_fc_z[sub] = np.arctanh(neut_net_res_fc[sub])

with np.load(f"{HCP_DIR}/atlas.npz") as dobj:
  atlas = dict(**dobj)

fear_net_res_fc_group = np.mean(fear_net_res_fc,axis=0)
neut_net_res_fc_group = np.mean(neut_net_res_fc,axis=0)

### background connectivity brain plot
# fear_net_plot = plotting.view_connectome(fear_net_res_fc_group, atlas["coords"], edge_threshold="99%")
# fear_net_plot.open_in_browser()
# neut_net_plot = plotting.view_connectome(neut_net_res_fc_group, atlas["coords"], edge_threshold="99%")
# neut_net_plot.open_in_browser()


mean_net_diff_map = np.mean(fear_net_res_fc_z,axis=0)-np.mean(neut_net_res_fc_z,axis=0)
# #create new rvs
z_net_map = -abs((mean_net_diff_map-0)/np.sqrt(1/N_SUBJECTS*(1/(N_FEAR_condition-3)+1/(N_NEUT_condition-3))))
# # z-test
# p_net_values = scipy.stats.norm.cdf(z_net_map,0,1)*2
#
# #FDR correction
# p_net_FDR_corrected=statsmodels.stats.multitest.fdrcorrection(p_net_values)[0].astype(int)

# FWE correction
# p_net_mask = p_net_values < 0.05/(12*12)
# p_net_cond = np.where(p_net_mask,p_net_values,-1)
# p_net_values_corrected = p_net_values[p_net_FDR_corrected]

fig = plt.gcf()
# cc_map = sns.heatmap(p_net_values, cmap='Blues', annot=False,xticklabels=list(network_names), yticklabels=list(network_names))
mask = np.zeros_like(z_net_map)
mask[np.triu_indices_from(mask)] = True
idx_th = np.where(abs(z_net_map)< 3)
mask[idx_th]= True

with sns.axes_style("white"):
    ax = sns.heatmap(z_net_map,mask=mask, annot=False,xticklabels=list(network_names), yticklabels=list(network_names))
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)

# sns.heatmap(z_net_map, annot=False, cmap="RdBu_r", xticklabels=list(network_names), yticklabels=list(network_names),ax=ax)
# cc_map.set_title('Network-based background conncetivity comparison')
# fig.set_size_inches(25, 20)
fig.savefig('network_BC_z',dpi = 200,bbox_inches='tight')
