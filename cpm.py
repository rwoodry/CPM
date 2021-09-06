from nilearn.regions import RegionExtractor
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
import nibabel as nib
import numpy as np
import pandas
import os
import glob
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from nilearn import plotting
from matplotlib import pyplot as plt
from nilearn import input_data
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
import scipy
import pickle
import seaborn
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
from tqdm import tqdm


def create_trial_labels(subjectnum, sync_pulses, test_trials):
    # Filter by subject
    test_labels = test_trials[test_trials['Subject'] == subjectnum].copy()
    
    # Initialize Pandas data series for Start and End TRs, as well as for fmri_run number
    start_TR = pandas.Series(dtype = 'float32')
    end_TR = pandas.Series(dtype = 'float32')
    fmri_run = pandas.Series(dtype = 'float32')
    
    # Get order of task type for labeling wich fmri run each entry corresponds too
    time_indices = np.unique(test_labels['times'], return_index = True)[1]
    task_order = []
    [task_order.append(test_labels.iloc[[i]]['Task_type'].to_string().split()[1]) for i in time_indices]
    

    
    
    for i in sync_pulses.index:

        data = test_labels[test_labels['Task_type'] == sync_pulses.trial_type[i]].copy()

        trial_times = np.unique
        start_tr = np.floor((data["trial_starttime"].copy() - sync_pulses.getready_response[i]) / 720).astype(int)
        end_tr = np.floor((data["trial_endtime"].copy() - sync_pulses.getready_response[i]) / 720).astype(int)
        run_num = task_order.index(sync_pulses.trial_type[i])


        start_TR = pandas.concat([start_TR, pandas.Series(start_tr)])
        end_TR = pandas.concat([end_TR, pandas.Series(end_tr)])
        
        if run_num > 1:
            run_num = 'Run' + str(run_num - 1)
        elif run_num == 1:
            run_num = 'Ex2' 
        else:
            run_num = 'Ex'
        fmri_run = pandas.concat([fmri_run, pandas.Series([run_num] * len(end_tr))])
        
        
        
        
    test_labels = test_labels[['Subject', 'Task_type', 'trial_starttime', 'trial_endtime', 'accuracy']]
    
    test_labels['start_TR'] = start_TR.values
    test_labels['end_TR'] = end_TR.values
    test_labels['fmri_run'] = fmri_run.values

    
    return test_labels, task_order


def subject_timeseries(mlindiv_filename, confounds_filename, atlas_filename):
    
    # Load fmri volumes
    func_img = nib.load(mlindiv_filename)
    header_zooms = func_img.header.get_zooms()
    TR = header_zooms[3]

    pandas.read_csv(confounds_filename, sep = '\t').fillna(0).to_csv("conf_nona_temp.tsv", sep = '\t', index = False)

    print('Atlas ROIs are located in nifti image (4D) at: %s' % atlas_filename)
    print('Func images are located in nifti image (4D) at: %s \n\t--- Confounds at: %s' % (mlindiv_filename, confounds_filename))
    print('Func images Voxel Dimensions (mm): %s\tFunc TR: %s' % (header_zooms[0:3], header_zooms[3]))

    # Create a masker from the HO atlas
    masker = input_data.NiftiMapsMasker(
        atlas_filename, 
        resampling_target='data',
        memory = 'nilearn_cache', 
        verbose = 0, low_pass=0.1, high_pass=0.01, detrend=True, t_r = TR).fit()


    # Create an overall time series for the run
    time_series = masker.transform(mlindiv_filename, confounds = 'conf_nona_temp.tsv')
    
    return time_series


def timewindows(timeseries, trial_labels, task_label, start_padding = 0, end_padding = 0):
    t_labels = trial_labels[trial_labels['Task_type'] == task_label]
    
    time_window_num = t_labels.shape[0]

    
    time_windows = []

    for i in range(time_window_num):
        
        start_TR = t_labels.start_TR.iloc[i] + start_padding
        end_TR = t_labels.end_TR.iloc[i] + end_padding
        
        time_window = timeseries[start_TR:end_TR, :]
        time_windows.append(time_window)
        
    return time_windows


def aggregate_timewindows(subject, func_directory, confounds_directory, trial_labels, atlas_filename, start_padding = 7, end_padding = 7):
    
    # Define list of files to be processed, given subject number and location of functional files.
    directory_path = "%s/sub-%s" % (func_directory, subject)
    trial_filenames = os.listdir(directory_path)
    print("Functional Images for Subject %s stored in %s" % (subject, directory_path))
    print("Functional files to be processed: " + str(trial_filenames))
    
    # Initialize empty Dictionary of Windowed - Time Series
    dict_timewindows = {}
    
    for run in trial_filenames:
        # Grab task label from matching the func run number to trial labels.
        run_name = run.split("bold")[1].split("_")[0]
        task_label = np.unique(trial_labels[trial_labels['fmri_run'] == run_name].Task_type)[0]
        print("Computing ROI Time Series for %s: %s" % (run_name, task_label))
        
        # Get path to func file and corresponding confounds file
        mlindiv_filename = directory_path+ '/' + run
        confounds_filename = glob.glob("%s/*%s*%s_*.tsv" % (confounds_directory, subject, run_name))[0]
        
        # Compute time series for func run
        time_series = subject_timeseries(mlindiv_filename, confounds_filename, atlas_filename)
        
        # Split time series into list time windows
        time_windows = timewindows(time_series, trial_labels, task_label, start_padding, end_padding)
        
        # For current Run, assign the list of time windows obtained from the previous line
        dict_timewindows[task_label] = time_windows
    
    return dict_timewindows


def model_connectome(ts_list, ts_labels, subject_set, C = 1, n_splits = 10, fconn_types = ['correlation', 'partial correlation', 'tangent', 'covariance', 'precision'], plot_scores = False):
    # If ts_list | ts_labels are a dictionary, collapse them into a list
    if type(ts_labels) == dict:
        ts_labels = {k: ts_labels[k] for k in ts_labels.keys() & subject_set}
        ts_list = {k: ts_list[k] for k in ts_list.keys() & subject_set}
        ts_labels = [item for sublist in list(ts_labels.values()) for item in sublist]
        ts_list = [item for sublist in list(ts_list.values()) for item in sublist]
        
    # Define and run the Model
    _, classes = np.unique(ts_labels, return_inverse=True)  # Convert accuracy into numpy array of binary labels
    ts_array = np.asarray(ts_list) # Convert list of time series into a numpy array of time series

    # Define correlation types
    kinds = fconn_types
    _, classes = np.unique(ts_labels, return_inverse=True)
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.3)
    ts_array = np.asarray(ts_list)

    scores = {}
    weights = {}
    print("RUNNING")
    if type(C) == list:
        for c in tqdm(C):
            scores[str(c)] = []
            for train, test in cv.split(ts_array, classes):
                connectivity = ConnectivityMeasure(kind = kind[0], vectorize = True) # We vectorize the Functional Connectivity, so it is easier to run the SVC classifier on
                connectomes = connectivity.fit_transform(ts_array[train])

                classifier = LinearSVC(C = c).fit(connectomes, classes[train])
                predictions = classifier.predict(connectivity.transform(ts_array[test]))
                scores[str(c)].append(accuracy_score(classes[test], predictions)) 
                weights[str(c)].append(np.array(classifier.coef_[0]))
    else:
        for kind in tqdm(kinds):
            scores[kind] = []
            for train, test in cv.split(ts_array, classes):
                connectivity = ConnectivityMeasure(kind = kind, vectorize = True) # We vectorize the Functional Connectivity, so it is easier to run the SVC classifier on
                connectomes = connectivity.fit_transform(ts_array[train])

                classifier = LinearSVC(C = C).fit(connectomes, classes[train])
                predictions = classifier.predict(connectivity.transform(ts_array[test]))
                scores[kind].append(accuracy_score(classes[test], predictions)) 
                weights[kind].append(classifier.coef_[0])
    
    if plot_scores:
        chance_level = np.mean(ts_labels)
        print('CHANCE: ', chance_level)
        mean_scores = [np.mean(scores[kind]) for kind in kinds]
        print('MEAN MODEL ACC: ', mean_scores)
        scores_std = [np.std(scores[kind]) for kind in kinds]
    
        plt.figure(figsize=(6, 4))
        positions = np.arange(len(kinds)) * .1 + .1
        plt.barh(positions, mean_scores, align='center', height=.05, xerr=scores_std)
        yticks = [k.replace(' ', '\n') for k in kinds]
        plt.yticks(positions, yticks)
        plt.gca().grid(True)
        plt.gca().set_axisbelow(True)
        plt.gca().axvline(chance_level, color='red', linestyle='--')
        plt.xlabel('Classification accuracy\n(red line = chance level)')
        plt.tight_layout()

    return(scores, weights)


def subject_timeseries_labels(sub_num_list, behav_directory, func_directory, confounds_directory, sync_pulses_directory, atlas_filename, output_to_filename_base):
    '''
    This function takes a list of subject numbers and runs through several helper functions iterating over each subject
    to output a dictionary of subject - timeseries pairs (X) and a dictionary of subject - labels pairs (Y) to be run through any model.
        Requirements:
            - A behavioral directory containing the trial master.csv
            - a func directory pointing to the functional files to be used. Make sure there are no duplicate runs
            - a confoudns directory contianing only conformatted .tsv files (refer to R script conformat)
            - a directory containing syncpulse.csv files for each subject
    It takes roughly 45 secs to extract and properly label each ROI time series per Test Scan. So for scans looking at 6 runs it can take a little under 4 min to fully run. 
    It does store extractions in a cache, however, so when rerun using previously run subjects it should be a lot quicker.
    '''
    # Read in trial data
    test_trials = pandas.read_csv('%s/MLINDIV_trial_master.csv' % behav_directory)
    
    subject_timeseries = {}
    subject_labels = {}
    errored_subjects = {}
    
    # Loop through each subject's and create ROI timeseries for each trial
    for sub_num in sub_num_list:
        try:
            # Read in the sync_pulse start time data
            sync_pulses = pandas.read_csv('%s/sub-%s_syncpulses.csv' % (sync_pulses_directory, sub_num))

            # Create trial labels for the subject, using the sync pulse trial start time info
            trial_labels, _ = create_trial_labels(int(sub_num), sync_pulses, test_trials)

            # Extract ROI time series into a trial dictionary (keys = Run name, values = list of ROI time series for that run) for each trial timewindow defined by the trial labels
            sub_TW = aggregate_timewindows(sub_num, func_directory, confounds_directory, trial_labels, atlas_filename)

            ts_list = []
            ts_labels = []

            # Iterate through Subject's Trial dictionary, creating a List of timeseries (ts_list) and a list of accuracy labels for those timeseries irrespective of run type
            for tasktype in sub_TW.keys():
                if not tasktype.isnumeric():      
                    acc = list(trial_labels[trial_labels['Task_type'] == tasktype]['accuracy'])
                    [ts_labels.append(value) for value in acc]
                    for ts in sub_TW[tasktype]:
                        ts_list.append(ts)   

            subject_timeseries[sub_num] = ts_list
            subject_labels[sub_num] = ts_labels
        except Exception as e:
            errored_subjects[sub_num] = e
    
    xname = 'X_%s.pickle' % output_to_filename_base
    yname = 'Y_%s.pickle' % output_to_filename_base
    
    with open(xname, 'wb') as handle:
        pickle.dump(subject_timeseries, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open(yname, 'wb') as y_handle:
        pickle.dump(subject_labels, y_handle, protocol = pickle.HIGHEST_PROTOCOL)
            
    print(errored_subjects)
    return subject_timeseries, subject_labels


def return_Xy_data(X_dict, Y_dict, subject_set, exclude_size = []):
    # If exclude_size is passed in the form of [min, max], remove all trial time windowed time series 
    # and corresponding trial accuracy labels from dictionary before training the model. DEfault is none
    ts_list = X_dict.copy()
    ts_labels = Y_dict.copy()
    
    if exclude_size:
        min_windowsize_cutoff = exclude_size[0]
        max_windowsize_cutoff = exclude_size[1]
        
        for subject in list(subject_set):
            for idx, trial in enumerate(ts_list[subject]):
                if trial.shape[0] < min_windowsize_cutoff or trial.shape[0] > max_windowsize_cutoff:
                    ts_list[subject].pop(idx)
                    ts_labels[subject].pop(idx)
                    
    # Initialize an empty list of subjects. Collapse X and y dicts into lists to feed into the model, 
    # while storing their subject IDs in the correct order.
    subjects = []
    if type(ts_labels) == dict:
        ts_labels = {k: ts_labels[k] for k in ts_labels.keys() & subject_set}
        ts_list = {k: ts_list[k] for k in ts_list.keys() & subject_set}
        subject_list = list(ts_labels.keys())
        
        ts_labels = [item for sublist in list(ts_labels.values()) for item in sublist]
        ts_list = [item for sublist in list(ts_list.values()) for item in sublist]
    
    # Create a list of len(y) that contains a subject ID repeated as many times as number of trials 
    # fed into the model corresponding to that subject.
    for sub in subject_set:

        subject_list = [sub] * len(Y_dict[sub])
        for subject in subject_list:
            subjects.append(subject)
        
    # Define and run the Model
    _, classes = np.unique(ts_labels, return_inverse=True)  # Convert accuracy into numpy array of binary labels
    ts_array = np.asarray(ts_list) # Convert list of time series into a numpy array of time series
    
    _, classes = np.unique(ts_labels, return_inverse=True)
    ts_array = np.asarray(ts_list)
    
    return ts_array, classes, subjects


def best_model(ts_array, classes, kind, C, n_splits):
    scores = []
    high_score = 0
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.3)
    count = -1
    acc_avg = 0
    n_features = X[0].shape[1] * (X[0].shape[1] - 1) / 2
    weights_avg = np.zeros(int(n_features))
    
    
    for train, test in cv.split(ts_array, classes):
            count += 1
            pct_complete = (count/n_splits) * 100
            print("%0.3f Percent Complete" % pct_complete,  end = "\r", flush = True)
            
            connectivity = ConnectivityMeasure(kind = kind, vectorize = True, discard_diagonal = True) # We vectorize the Functional Connectivity, so it is easier to run the SVC classifier on
            connectomes = connectivity.fit_transform(ts_array[train])

            classifier = LinearSVC(C = C).fit(connectomes, classes[train])
            predictions = classifier.predict(connectivity.transform(ts_array[test]))
            acc_score = accuracy_score(classes[test], predictions)
            scores.append(acc_score)
            weights_avg += classifier.coef_[0]
            acc_avg += acc_score
            
            if acc_score > high_score:
                best_model = classifier
                high_score = acc_score
                
    weights_avg /= n_splits
    acc_avg /= n_splits
    
    
    return scores, best_model, high_score, weights_avg, acc_avg
            

def create_binned_performance(X, y, model, connectivity, subjects):
    window_size, true_accuracy, pred_accuracy, correct_prediction = [], [], [], []
    
    for index, connectome in enumerate(X):
        window_size.append(connectome.shape[0])
        true_accuracy.append(y[index])
        pred_accuracy.append(best_model.predict(connectivity.transform([X[index]]))[0])
        correct_prediction.append(y[index] == best_model.predict(connectivity.transform([X[index]]))[0])
        print("\r %03f Percent Complete" % (index/len(y)*100), end = "")
    
    d = {'subject': subjects, 'window_size': window_size, 'true_acc': true_accuracy, 'pred_acc': pred_accuracy, 'model_correct': correct_prediction}
    best_model_df = pandas.DataFrame(d)
    return best_model_df


