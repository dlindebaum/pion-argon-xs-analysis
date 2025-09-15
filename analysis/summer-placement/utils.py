"""
Utils package for pi0 analysis.

This package contains utility functions organized into modules:
- master: Main ML selection function
- data_processing: Data cleaning and splitting functions
- data_extraction: Data extraction from ntuple files
- metrics: Performance metrics calculations
- general: General utility functions
- hp_tuning: Hyperparameter tuning functions
- cut_approach: Cut-based selection functions
- plotting: Plotting and visualization functions
"""
import numpy as np
import pandas as pd
import pickle
import awkward as ak
import os
import glob
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import xgboost
from python.analysis import Master, SelectionTools, Plots
from python.analysis.Tags import GenerateTrueParticleTagsPiPlus

# =============================================================================
# GENERAL UTILITY FUNCTIONS
# =============================================================================

def find_particle_from_tags(tags, event, track):
    """Find particle type from tags for a given event and track."""
    for k, v in tags.items():
        if v.mask[event][track]:
            return k

def combine_fiducial_and_beam_masks(fiducial_mask, beam_mask):
    """Combine fiducial and beam masks."""
    i = -1
    combined_mask = []

    for event in range(len(fiducial_mask)):
        if fiducial_mask[event] is True:
            i += 1
            if i == len(beam_mask):
                return combined_mask
            if beam_mask[i] is True:    
                combined_mask.append(True)
            else:
                combined_mask.append(False)
        else:
            combined_mask.append(False)
    return combined_mask

def find_distance(pos_1, pos_2):
    """Find the distance between two 3D position vectors."""
    return np.sqrt((pos_1["x"] - pos_2["x"])**2 + (pos_1["y"] - pos_2["y"])**2 + (pos_1["z"] - pos_2["z"])**2)

def count_instances_in_list(list, X : list):
    """Count the number of particles of a given type (or set of types) in a list."""
    count = 0
    for item in list:
        if item in X:
            count += 1
    return count

def classify_X_as_Y(list_a, list_b, X : list, Y : list):
    """Count the number of times a particle of type X is identified as a particle of type Y."""
    X_as_Y = 0
    if type(list_a) != list:
        list_a = list_a.tolist()
    if type(list_b) != list:
        list_b = list_b.tolist()
    for i in range(len(list_a)):
        if list_a[i] in X and list_b[i] in Y:
            X_as_Y += 1
    return X_as_Y

def create_confusion_matrix(y_test, y_pred, dropped_particles=None):
    """Create confusion matrix with purity and efficiency information."""
    labels = sorted(list(set(y_test) | set(y_pred)))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    purities = []
    efficiencies = []
    counts = cm.flatten()

    for true_particle in range(len(labels)):
        for predicted_particle in range(len(labels)):
            purity_ = f"{purity(y_pred, y_test, [labels[predicted_particle]], [labels[true_particle]]):.3f}"
            efficiency_ = f"{efficiency(y_pred, y_test, [labels[predicted_particle]], [labels[true_particle]], dropped_particles):.3f}"
            purities.append(purity_)
            efficiencies.append(efficiency_)

    info = zip(counts, purities, efficiencies)
    info = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in info]
    info = np.asarray(info).reshape(cm.shape)

    cm = cm[::-1]
    info = info[::-1]

    return cm, info, labels

def get_classified_df(x_test, y_test, y_pred, class_label, return_before_classification=False):
    """Get dataframe with classified particles of the class: class_label."""
    df_before_classification = x_test.copy()
    df_before_classification["particle"] = y_test
    mask = np.array(y_pred) == class_label
    df_after_classification = df_before_classification.iloc[mask]

    if return_before_classification:
        return df_after_classification, df_before_classification
    else:
        return df_after_classification

# =============================================================================
# METRICS FUNCTIONS
# =============================================================================

def calculate_uncertainty(k, n):
    """Calculate uncertainty for a metric."""
    metric = k / n
    uncertainty = np.sqrt(metric * (1 - metric) / n)
    return uncertainty

def purity(y_pred, y_test, identified_particles : list, true_particles : list, return_uncertainty=False):
    """Calculate purity - ratio of correctly identified particles to total identified particles."""
    matched = classify_X_as_Y(y_pred, y_test, identified_particles, true_particles)
    total_identified = count_instances_in_list(y_pred, identified_particles)

    if total_identified == 0:
        return 0
    else:
        if return_uncertainty:
            return matched / total_identified, calculate_uncertainty(matched, total_identified)
        else:
            return matched / total_identified

def efficiency(y_pred, y_test, identified_particles : list, true_particles : list, dropped_particles : list = None, return_uncertainty=False):
    """Calculate efficiency - ratio of correctly identified particles to total real particles."""
    matched = classify_X_as_Y(y_pred, y_test, identified_particles, true_particles)
    total_real = count_instances_in_list(y_test, true_particles)
    if dropped_particles is not None:
        total_real += count_instances_in_list(dropped_particles, true_particles)
    if total_real == 0:
        return 0
    else:
        if return_uncertainty:
            return matched / total_real, calculate_uncertainty(matched, total_real)
        else:
            return matched / total_real

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def split_data(df, test_size=0.2, random_state=42, verbose=False, binary_classification=False):
    """Split the dataframe into training and testing sets."""
    x = df.drop(["particle"], axis=1)
    y = df["particle"]

    # transform to binary classification
    if binary_classification:
        y = convert_to_binary(y, "$\pi^{\pm}$")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    if verbose:
        print(f"Training set size: {len(x_train)} tracks.")
        print(f"Testing set size: {len(x_test)} tracks.")

    return x_train, x_test, y_train, y_test

def clean_df(df, return_dropped=False, verbose=False):
    """Clean the dataframe by removing rows with missing entries."""
    removed = 0
    rows_to_remove = []
    for i in range(len(df["track_dEdX_mean"])):
        if df["track_dEdX_mean"].isnull().values[i] == True:
            removed += 1
            rows_to_remove.append(i)

    dropped_rows = df.iloc[rows_to_remove]
    df = df.drop(rows_to_remove, axis=0)
    df = df.reset_index(drop=True)

    if verbose:
        output = f"Removed {removed} rows with missing entries. "
        print(output)

    if return_dropped:
        return df, dropped_rows
    else:
        return df

def convert_to_binary(y, particle_type="$\pi^{\pm}$"):
    """Convert the particle type to a binary classification."""
    for i in range(len(y)):
        if y[i] == "$\pi^{\pm}$":
            y[i] = 1
        else:
            y[i] = 0
    return y

# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================

def get_mc_masks(mc_file):
    """Get the mc masks from the ntuple file."""
    fiducial_masks = Master.LoadObject("/home/pemb6649/pi0-analysis/analysis/work/analysis_full/masks_mc/fiducial_selection_masks.dill")
    beam_masks = Master.LoadObject("/home/pemb6649/pi0-analysis/analysis/work/analysis_full/masks_mc/beam_selection_masks.dill")
    pi_masks = Master.LoadObject("/home/pemb6649/pi0-analysis/analysis/work/analysis_full/masks_mc/pi_selection_masks.dill")

    fiducial_mask = fiducial_masks[mc_file]
    beam_mask = beam_masks[mc_file]
    pi_mask = pi_masks[mc_file]
    
    combined_fiducial_mask = SelectionTools.CombineMasks(fiducial_mask)
    combined_beam_mask = SelectionTools.CombineMasks(beam_mask)
    combined_pi_mask = SelectionTools.CombineMasks(pi_mask)

    return combined_fiducial_mask, combined_beam_mask, combined_pi_mask

def get_mc_data(mc_file, combine_prior_mask=True, verbose=False):
    """Get the mc data and masks from the ntuple file."""
    mc = Master.Data(mc_file, nTuple_type = Master.Ntuple_Type.PDSP, target_momentum = 2)
    fiducial_mask, beam_mask, pi_mask = get_mc_masks(mc_file)

    prior_mask = combine_fiducial_and_beam_masks(fiducial_mask, beam_mask)

    pion_selection = 0
    for event in pi_mask:
        pion_selection += sum(event)

    if verbose:
        print(f"Original events: {len(mc.eventNum)}")
        print(f"Events after beam selection: {sum(prior_mask)}")
        print(f"PFOs classified as pions: {pion_selection}")

    if combine_prior_mask:
        return mc, prior_mask, pi_mask
    else:
        return mc, fiducial_mask, beam_mask, pi_mask

def save_file(data, output_path):
    """Save the data to a pickle file."""
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {data} to {output_path}")

def load_file(file_path):
    """Load the data from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

def extract_data(mc_files, size_per_file=-1, data_path=None, verbose=False, save=False):
    """Get all the data from the ntuple file."""
    all_data = []
    total_events = 0
    total_events_after_beam_selection = 0
    for i, mc_file in enumerate(mc_files):
        file_name = mc_file.split("/")[-1]
        if verbose:
            print("-"*85)
            print(f"Processing [{i+1}/{len(mc_files)}]: {file_name}")
            print("-"*85)
        mc, combined_prior_mask, combined_pi_mask = get_mc_data(mc_file, verbose=False)
        data = extract_observables(mc, size=size_per_file, beam_selection_mask=combined_prior_mask, verbose=True, use_custom_observables=True)
        all_data.extend(data)
        total_events += len(mc.eventNum)
        total_events_after_beam_selection += sum(combined_prior_mask)

    if verbose:
        print(f"Total events: {total_events}")
        print(f"Total events after beam selection: {total_events_after_beam_selection}")

    if save:
        if data_path is None:
            data_path = "/home/pemb6649/pi0-analysis/analysis/summer-placement/extracted-data/all_data.pkl"
        save_file(all_data, data_path)

def get_length_from_file(file_path):
    """Get the length of the data."""
    data = load_file(file_path)
    return len(data)

def extract_observables(mc, size=-1, start=0, beam_selection_mask=None, verbose=False, use_custom_observables=False):
    """Extract the observables from the ntuple."""
    if size == -1 or size > len(mc.recoParticles.track_chi2_proton):
        num_events = len(mc.recoParticles.track_chi2_proton)
    else:
        num_events = size

    track_data = []
    tags = GenerateTrueParticleTagsPiPlus(mc)

    skipped_tracks = 0
    rejected_events = 0

    for event in range(start, num_events):
        if beam_selection_mask is not None:
            if beam_selection_mask[event] is False:
                rejected_events += 1
                continue

        for track in range(len(mc.recoParticles.track_chi2_proton[event])):
            try:
                individual_track_info = {
                    # calorimeter information
                    "track_dEdX_mean": np.mean(mc.recoParticles.track_dEdX[event][track]),
                    "track_dEdX_median": np.median(mc.recoParticles.track_dEdX[event][track]),
                    "track_dQdX_mean": np.mean(mc.recoParticles.track_dQdX[event][track]),
                    "track_dQdX_median": np.median(mc.recoParticles.track_dQdX[event][track]),
                    "residual_range_median": np.median(mc.recoParticles.residual_range[event][track]),
                    "residual_range_mean": np.mean(mc.recoParticles.residual_range[event][track]),
                    
                    # track information
                    "track_chi2/ndof_proton": mc.recoParticles.track_chi2_proton[event][track] / mc.recoParticles.track_chi2_proton_ndof[event][track],
                    "track_chi2/ndof_pion": mc.recoParticles.track_chi2_pion[event][track] / mc.recoParticles.track_chi2_pion_ndof[event][track],
                    "track_chi2/ndof_muon": mc.recoParticles.track_chi2_muon[event][track] / mc.recoParticles.track_chi2_muon_ndof[event][track],
                    "track_length": mc.recoParticles.track_len[event][track],
                    "track_score": mc.recoParticles.track_score[event][track],

                    # vertex information
                    "track_vertex_michel": mc.recoParticles.track_vertex_michel[event][track],
                    "track_vertex_nhits": mc.recoParticles.track_vertex_nhits[event][track],

                    # geometric information
                    "track_start_pos_x": mc.recoParticles.track_start_pos[event][track]["x"],
                    "track_start_pos_y": mc.recoParticles.track_start_pos[event][track]["y"],
                    "track_start_pos_z": mc.recoParticles.track_start_pos[event][track]["z"],
                    "track_end_pos_x": mc.recoParticles.track_end_pos[event][track]["x"],
                    "track_end_pos_y": mc.recoParticles.track_end_pos[event][track]["y"],
                    "track_end_pos_z": mc.recoParticles.track_end_pos[event][track]["z"],

                    # truth information
                    "particle": find_particle_from_tags(tags, event, track),
                }
                if use_custom_observables:
                    custom_observables = {
                        "hit_density": mc.recoParticles.n_hits_collection[event][track] / mc.recoParticles.track_len[event][track],
                        "angle_to_beam": np.dot(np.array(list(mc.recoParticles.beam_startPos[event].to_list().values())) - np.array(list(mc.recoParticles.beam_endPos[event].to_list().values())), np.array(list(mc.recoParticles.track_start_dir[event][track].to_list().values()))),
                        "distance_from_beam": find_distance(mc.recoParticles.track_start_pos[event][track], mc.recoParticles.beam_endPos[event]),
                        "dEdX/dQdX": np.mean(mc.recoParticles.track_dEdX[event][track]) / np.mean(mc.recoParticles.track_dQdX[event][track]),
                        "dEdX_per_length": np.mean(mc.recoParticles.track_dEdX[event][track]) / mc.recoParticles.track_len[event][track],
                        "angle_to_vertical": np.dot(np.array(list(mc.recoParticles.track_start_dir[event][track].to_list().values())), np.array([0,0,1])),
                        "dEdX_end_pos": np.mean(mc.recoParticles.track_dEdX[event][track][-3:-1])
                    }
                    individual_track_info.update(custom_observables)

            except Exception as e:
                skipped_tracks += 1
                continue

            track_data.append(individual_track_info)

    if verbose:
        print(f"Considered {num_events} original events.")
        print(f"Rejected {rejected_events} events through beam selection.")
        print(f"Skipped {skipped_tracks} tracks due to errors.")
        print(f"Number of tracks: {len(track_data)}")

    return track_data

def apply_track_operation(data, operation):
    """Apply an operation to each track in the data."""
    return ak.Array([[operation(track) for track in event] for event in data])

# =============================================================================
# CUT-BASED APPROACH FUNCTIONS
# =============================================================================

def convert_mask_to_y(tags, fiducial_mask, beam_mask, pi_mask, size=-1):
    """Convert the masks to a binary classification for pions."""
    y_test = []
    y_pred = []

    j = -1
    k = -1
    if size == -1 or size > len(fiducial_mask):
        size = len(fiducial_mask)
    for event in range(size):
        if fiducial_mask[event] is False:
            continue
        j += 1
        if beam_mask[j] is False:
            continue
        k += 1
        for track in range(len(pi_mask[k])):
            if find_particle_from_tags(tags, event, track) == "$\pi^{\pm}$":
                y_test.append(1)
            else:
                y_test.append(0)
            if pi_mask[k][track]:
                y_pred.append(1)
            else:
                y_pred.append(0)
    
    return y_test, y_pred

def cut_based_selection(mc_files, size=-1, verbose=False, plot=False):
    """Measure the performance of the cut-based selection using the masks."""
    y_test = []
    y_pred = []

    for mc_file in mc_files:
        mc, fiducial_mask, beam_mask, pi_mask = get_mc_data(mc_file, combine_prior_mask=False, verbose=False)
        tags = GenerateTrueParticleTagsPiPlus(mc)
        y_test_i, y_pred_i = convert_mask_to_y(tags, fiducial_mask, beam_mask, pi_mask, size)
        y_test.extend(y_test_i)
        y_pred.extend(y_pred_i)
    
    if verbose:
        pion_purity, pion_purity_uncertainty = purity(y_pred, y_test, [1], [1], return_uncertainty=True)
        pion_efficiency, pion_efficiency_uncertainty = efficiency(y_pred, y_test, [1], [1], return_uncertainty=True)
        print(f"Pion purity: {pion_purity*100:.1f}% ± {pion_purity_uncertainty*100:.1f}%")
        print(f"Pion efficiency: {pion_efficiency*100:.1f}% ± {pion_efficiency_uncertainty*100:.1f}%")

    if plot:
        cm, info, labels= create_confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, info, labels, plot=plot, size=(10, 8))

# =============================================================================
# HYPERPARAMETER TUNING FUNCTIONS
# =============================================================================

def tune_hp(hp_tuning, data, model_name, plot=False, save=True, verbose=False, binary_classification=False, consider_dropped_rows=False):
    """Tune a hyper parameter (hp) for a given model."""
    hp_tuning_results = {}
    if model_name == "rf":
        full_name = "Random Forest"
    elif model_name == "xgb":
        full_name = "XGBoost"
    elif model_name == "mlp":
        full_name = "MLP"
    elif model_name == "dt":
        full_name = "Decision Tree"
    else:
        raise ValueError(f"Model {model_name} not supported.")
        return

    print(f"Tuning {full_name} HYPERPARAMETERS".upper())
    for hp_name, hp_values in hp_tuning.items():
        
        pion_purities = []
        pion_efficiencies = []
        
        for value in hp_values:
            
            if verbose:
                print("--------------------------------")
                print(f"Testing {hp_name} = {value}...")

            if model_name == "rf":
                model = RandomForestClassifier(random_state=42, **{hp_name: value})
            elif model_name == "xgb":
                model = XGBClassifier(random_state=42, **{hp_name: value})
            elif model_name == "mlp":
                model = MLPClassifier(random_state=42, **{hp_name: value})
            elif model_name == "dt":
                model = DecisionTreeClassifier(random_state=42, **{hp_name: value})
            else:
                raise ValueError(f"Model {model_name} not supported.")
                return

            model, pion_purity, pion_efficiency = master_pion_selection(data=data, model=model, verbose=False, plot=False, binary_classification=binary_classification, consider_dropped_rows=consider_dropped_rows, return_performance=True);
            print(f"Purity: {pion_purity:.3f}, Efficiency: {pion_efficiency:.3f}")
            pion_purities.append(pion_purity)
            pion_efficiencies.append(pion_efficiency)

        if plot:
            plot_tuning_results(hp_name, hp_values, pion_purities, pion_efficiencies)

        hp_tuning_results[hp_name] = (hp_values, pion_purities, pion_efficiencies)

    if save:
        save_hp_tuning_results(hp_tuning_results, model_name)
        
    return hp_tuning_results

def save_hp_tuning_results(hp_tuning_results, model_name):
    """Save hyperparameter tuning results to file."""
    output_dir = "/home/pemb6649/pi0-analysis/analysis/summer-placement/hp-tuning-results/"
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}{model_name}_{current_time}.pkl"

    with open(filename, "wb") as file:
        pickle.dump(hp_tuning_results, file)

def find_best_hps(hp_tuning_results):
    """Find the best hyperparameters from tuning results."""
    pass

# =============================================================================
# MASTER FUNCTION
# =============================================================================

def master_pion_selection(data=None, data_path=None, ntuple=None, size=-1, model=None, 
                        custom_observables=None, verbose=False, plot=False, 
                        save=False, binary_classification=False, drop_cols=None, 
                        consider_dropped_rows=False, 
                        title="", overfit_test=False):
    """Master function to run the ML selection."""
    # extract observables if not provided
    if data is None and data_path is None:
        if ntuple is None:
            raise ValueError("Either data or ntuple must be provided.")
        data = extract_observables(ntuple, size)
    elif data_path is not None:
        data = load_file(data_path)

    if binary_classification:
        title += f", binary classification, {len(data)} events"
    else:
        title += f", full classification, {len(data)} events"

    # add custom observables if provided
    combined_data = []
    if custom_observables is not None:
        for i in range(len(custom_observables)):
            combined_data.append(data[i] | custom_observables[i])
        data = combined_data
        
    df = pd.DataFrame(data)
    df = clean_df(df, return_dropped=False, verbose=verbose)

    if drop_cols is not None:
        df.drop(columns=drop_cols, inplace=True)
        
    if binary_classification:
        x_train, x_test, y_train, y_test = split_data(df, verbose=verbose, binary_classification=True)
        pions = "1"
        not_pions = "0"
    else:
        x_train, x_test, y_train, y_test = split_data(df, verbose=verbose, binary_classification=False)
        pions = "$\pi^{\pm}$"

    if model is None:
        model = RandomForestClassifier(random_state=42)

    model, le = train_model(model, x_train, y_train, verbose=verbose)

    y_test_pred = get_predictions(model, le, x_test)
    results_df = get_class_performance(model, le, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, class_label=pions)

    if verbose:
        print(results_df)
    
    if not save and not plot:
        return model

    df_after_classification, df_before_classification = get_classified_df(x_test, y_test, y_test_pred, pions, return_before_classification=True)

    if save:
        performance_dir = "/home/pemb6649/pi0-analysis/analysis/summer-placement/model-performance"
        existing_files = glob.glob(os.path.join(performance_dir, "performance_*.pdf"))
        max_num = 0
        for f in existing_files:
            basename = os.path.basename(f)
            try:
                num = int(basename.split("_")[1].split(".")[0])
                if num > max_num:
                    max_num = num
            except (IndexError, ValueError):
                continue
        next_num = max_num + 1
        performance_filename = os.path.join(performance_dir, f"performance_{next_num}.pdf")
        book = Plots.PlotBook(performance_filename)
    else:
        book = None

    title = f"{title}\n{datetime.now().strftime('%Y/%m/%d | %H:%M')}"

    plot_df(results_df, title=title , size=(8, 4), book=book, plot=plot)
    plot_feature_importances(model, x_train, book=book, plot=plot, size=(14, 8))
    
    cm, info, labels = create_confusion_matrix(y_test, y_test_pred, None)
    plot_confusion_matrix(cm, info, labels, book=book, plot=plot, size=(12, 10))

    plot_correlation_matrix(df_before_classification, title="Correlation Matrix of Observables BEFORE classification", size=(15, 12), book=book, plot=plot)
    plot_correlation_matrix(df_after_classification, title="Correlation Matrix of Observables AFTER classification", size=(15, 12), book=book, plot=plot)
    plot_observables_grid(df_before_classification, title="Observables BEFORE classification", percentiles=[0.1, 0.9], norm=True, book=book, plot=plot, size=(26, 28), binary_classification=binary_classification)
    plot_observables_grid(df_after_classification, title="Observables AFTER classification", percentiles=[0.1, 0.9], norm=True, book=book, plot=plot, size=(26, 28), binary_classification=binary_classification)

    if save:
        book.close()

    return model

def get_class_performance(model, le=None, dtrain=None, dtest=None, x_test=None, y_test=None, x_train=None, y_train=None, class_label=None, threshold=0.5):
    """Get the performance of a model for a given class."""
    if dtrain is not None and dtest is not None:
        y_train_pred = get_predictions(model, le, x_test=dtrain, threshold=threshold)
        y_test_pred = get_predictions(model, le, x_test=dtest, threshold=threshold)

        y_train = dtrain.get_label()
        y_test = dtest.get_label()

        if le is not None:
            # DMatrix objects for full classification were defined with encoded labels
            y_train = le.inverse_transform(y_train.astype(int))
            y_test = le.inverse_transform(y_test.astype(int))
        
    else:
        y_train_pred = get_predictions(model, le, x_test=x_train)
        y_test_pred = get_predictions(model, le, x_test=x_test)

    if class_label == 1:
        # safe way to ensure for binary classification (0 or 1) the outputs are integers and not strings e.g. "0" and "1"
        y_train_pred = y_train_pred.astype(int)
        y_test_pred = y_test_pred.astype(int)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

    purity_train, purity_uncertainty_train = purity(y_train_pred, y_train, [class_label], [class_label], return_uncertainty=True)
    efficiency_train, efficiency_uncertainty_train = efficiency(y_train_pred, y_train, [class_label], [class_label], return_uncertainty=True)
    purity_test, purity_uncertainty_test = purity(y_test_pred, y_test, [class_label], [class_label], return_uncertainty=True)
    efficiency_test, efficiency_uncertainty_test = efficiency(y_test_pred, y_test, [class_label], [class_label], return_uncertainty=True)

    results_dict = {
        f"{class_label}": [
            f"Test Particles",
            "Purity",
            "Efficiency",
            "Purity x Efficiency"
        ],
        "Test Data": [
            len(y_test),
            f"{purity_test*100:.1f}% ± {purity_uncertainty_test*100:.1f}%",
            f"{efficiency_test*100:.1f}% ± {efficiency_uncertainty_test*100:.1f}%",
            f"{purity_test * efficiency_test*100:.1f}% ± {purity_uncertainty_test*100 + efficiency_uncertainty_test*100:.1f}%"
        ],
        "Train Data": [
            len(y_train),
            f"{purity_train*100:.1f}% ± {purity_uncertainty_train*100:.1f}%",
            f"{efficiency_train*100:.1f}% ± {efficiency_uncertainty_train*100:.1f}%",
            f"{purity_train * efficiency_train*100:.1f}% ± {purity_uncertainty_train*100 + efficiency_uncertainty_train*100:.1f}%"
        ]
    }
    results_df = pd.DataFrame(results_dict)
    return results_df

def train_model(model, x_train, y_train, verbose=False, encode_labels=True):

    if encode_labels:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

    if verbose:
        print("Fitting model...")

    model.fit(x_train, y_train);

    if verbose:
        print("Model fitted.")
    
    if encode_labels:
        return model, le
    else:
        return model

def get_predictions(model, le=None, x_test=None, threshold=0.5):
    y_pred = model.predict(x_test);
    if type(x_test) == xgboost.core.DMatrix:  
        # if working with lower level XGBoost, model.predict gives class probabilities for each sample (like predict_proba for sklearn)
        try:
            # multiclass classification -> take the class with the highest probability
            y_pred = y_pred.argmax(axis=1)
        except:
            try:
                # binary classification -> 1 only if the probability > threshold, else 0
                y_pred = (y_pred > threshold).astype(int)
            except:
                raise ValueError("Threshold too high! Try a lower threshold.")
    if le is not None:
        y_pred = le.inverse_transform(y_pred)
    return y_pred

def get_class_probabilities(model, le=None, y_proba=None, x_test=None, class_label=None):
    if y_proba is None:
        if type(x_test) == xgboost.core.DMatrix:
            y_proba = model.predict(x_test);    
        else:
            y_proba = model.predict_proba(x_test);
    if le is not None:
        class_index = le.transform([class_label])[0]
    else:
        class_index = class_label
    class_proba = y_proba[:, class_index]
    return class_proba

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_df(df, title=None, size=(20, 15), book=None, plot=False, 
           style='modern', colormap='viridis', header_color=None, 
           cell_color=None, text_color='black', font_size=10,
           title_font_size=16, border_width=1.5):
    
    # Create figure with better DPI for crisp text
    fig, ax = plt.subplots(figsize=size, dpi=100)
    ax.axis('off')
    
    # Style configurations
    styles = {
        'modern': {
            'header_color': '#2E86AB' if header_color is None else header_color,
            'cell_colors': ['#F8F9FA', '#E9ECEF'] if cell_color is None else [cell_color, cell_color],
            'text_color': text_color,
            'header_text_color': 'white',
            'edge_color': '#DEE2E6'
        },
        'classic': {
            'header_color': '#8B4513' if header_color is None else header_color,
            'cell_colors': ['#FFF8DC', '#F5F5DC'] if cell_color is None else [cell_color, cell_color],
            'text_color': text_color,
            'header_text_color': 'white',
            'edge_color': '#D2B48C'
        },
        'dark': {
            'header_color': '#1F2937' if header_color is None else header_color,
            'cell_colors': ['#374151', '#4B5563'] if cell_color is None else [cell_color, cell_color],
            'text_color': 'white' if text_color == 'black' else text_color,
            'header_text_color': 'white',
            'edge_color': '#6B7280'
        },
        'minimal': {
            'header_color': '#FFFFFF' if header_color is None else header_color,
            'cell_colors': ['#FFFFFF', '#F9F9F9'] if cell_color is None else [cell_color, cell_color],
            'text_color': text_color,
            'header_text_color': 'black',
            'edge_color': '#E5E5E5'
        }
    }
    
    current_style = styles.get(style, styles['modern'])
    
    # Create the table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.8/len(df.columns)] * len(df.columns)  # Dynamic column width
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 2)  # Make rows taller for better readability
    
    # Style header row
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor(current_style['header_color'])
        cell.set_text_props(weight='bold', color=current_style['header_text_color'])
        cell.set_edgecolor(current_style['edge_color'])
        cell.set_linewidth(border_width)
    
    # Style data rows with alternating colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            # Alternate row colors
            color_idx = (i - 1) % 2
            cell.set_facecolor(current_style['cell_colors'][color_idx])
            cell.set_text_props(color=current_style['text_color'])
            cell.set_edgecolor(current_style['edge_color'])
            cell.set_linewidth(border_width)
    
    # Add title with better formatting
    if title:
        plt.suptitle(title, fontsize=title_font_size, fontweight='bold', 
                    y=0.98, color=current_style['text_color'])
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9 if title else 0.95)
    
    # Handle book saving
    if book is not None:
        book.Save()
    
    # Handle plot display
    if not plot:
        plt.close()
    else:
        plt.show()

def plot_tuning_results(hp_name, hp_values, pion_purities, pion_efficiencies):
    """Plot the purity and efficiency tuning for a given hyper parameter (hp)."""
    plt.plot(hp_values, pion_purities, label="purity")
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.plot(hp_values, pion_efficiencies, label="efficiency")
    plt.xlabel(hp_name)
    plt.ylabel("Purity/Efficiency")
    plt.title(f"Purity/Efficiency tuning for {hp_name}")
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def plot_observables_grid(df, percentiles=[0.1, 0.9], ncols=4, size=(20, 15), norm=False, book=None, plot=False, title=None, binary_classification=False):
    """Create a grid of plots for multiple observables using a pandas DataFrame."""
    if binary_classification:
        particle_colours = {
            "1": "#66c2a5",      # teal
            "0": "#fc8d62",      # orange
        }
    else:
        particle_colours = {
            "$\mu^{\pm}$": "#66c2a5",      # teal
            "$\pi^{\pm}$": "#fc8d62",      # orange
            "$\pi^{\pm}$:2nd": "#8da0cb",  # blue
            "$p$": "#e78ac3",              # pink
            "other": "#a6d854",            # green
            "$e^{+}$": "#ffd92f",          # yellow
            "$\gamma$": "#e5c494"          # beige
        }

    # Exclude the 'particle' column from observables to plot
    obs_names = [col for col in df.columns if col != "particle"]
    n_obs = len(obs_names)
    nrows = (n_obs + ncols - 1) // ncols  # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=size)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.flatten()

    # Get unique particle tags in the order they appear
    tags = list(df["particle"].unique())

    # Build a list of colours for the tags, using the dictionary, fallback to a default if not found
    default_colour = "#bbbbbb"
    tag_colours = [particle_colours.get(str(tag), default_colour) for tag in tags]

    for i, obs_name in enumerate(obs_names):
        ax = axes_flat[i]

        # Split data by tags
        split_data = [df.loc[df["particle"] == tag, obs_name] for tag in tags]

        # Concatenate all data for x-range calculation
        all_data = np.concatenate([d for d in split_data])

        lower = np.percentile(all_data, percentiles[0]*100)
        upper = np.percentile(all_data, percentiles[1]*100)
        x_range = (lower, upper)

        # Plot stacked histogram with consistent colours
        ax.hist(
            split_data,
            bins=30,
            range=x_range,
            stacked=True,
            label=[str(tag).replace("_", " ") for tag in tags],
            density=norm,
            color=tag_colours
        )

        ax.set_xlabel(obs_name.replace('_', ' '))
        ax.set_ylabel('Normalized entries' if norm else 'Entries')
        ax.set_title(obs_name.replace('_', ' '))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Add space at the top for the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Hide unused subplots
    for i in range(n_obs, len(axes_flat)):
        axes_flat[i].set_visible(False)

    if title is not None:
        fig.suptitle(title, fontsize=18)

    if book is not None:
        book.Save()

    if not plot:
        plt.close()

def plot_correlation_matrix(df, obs_to_plot=None, size=(20, 15), title=None, book=None, plot=False):
    """Plot a correlation matrix of the observables using seaborn."""
    if obs_to_plot is None:
        obs_to_plot = [col for col in df.columns if col != "particle"]

    lower = df[obs_to_plot].quantile(0.1)
    upper = df[obs_to_plot].quantile(0.9)

    # Build a mask: keep only rows where all obs_to_plot are within [1st, 99th] percentiles
    mask = np.ones(len(df), dtype=bool)
    for obs in obs_to_plot:
        mask &= (df[obs] >= lower[obs]) & (df[obs] <= upper[obs])

    df_filtered = df[mask].copy()

    if obs_to_plot is None:
        correlation_matrix = df_filtered.corr()
    else:
        correlation_matrix = df_filtered[obs_to_plot].corr()

    plt.figure(figsize=size)
    sns.heatmap(correlation_matrix, 
                annot=True,
                cmap='RdBu_r',  # nice colour scheme
                center=0,  # center the colour scheme at 0 i.e. 0 = white
                square=True,
                fmt='.2f')

    plt.title(title)

    if book is not None:
        book.Save()

    if not plot:
        plt.close()

def plot_scatter_matrix(df, obs_to_plot=None, title=None, size=(10, 8), save=False, plot=False, binary_classification=False):
    """Plot a scatter matrix of the observables using matplotlib."""
    if binary_classification:
        particle_colours = {
            "1": "#66c2a5",      # teal
            "0": "#fc8d62",      # orange
        }
    else:
        particle_colours = {
            "$\mu^{\pm}$": "#66c2a5",      # teal
            "$\pi^{\pm}$": "#fc8d62",      # orange
            "$\pi^{\pm}$:2nd": "#8da0cb",  # blue
            "$p$": "#e78ac3",              # pink
            "other": "#a6d854",            # green
            "$e^{+}$": "#ffd92f",          # yellow
            "$\gamma$": "#e5c494"          # beige
        }

    # Compute 10th and 90th percentiles for each observable
    if obs_to_plot is None:
        obs_to_plot = [col for col in df.columns if col != "particle"]
    lower = df[obs_to_plot].quantile(0.1)
    upper = df[obs_to_plot].quantile(0.9)

    # Build a mask: keep only rows where all obs_to_plot are within the percentiles
    mask = np.ones(len(df), dtype=bool)
    for obs in obs_to_plot:
        mask &= (df[obs] >= lower[obs]) & (df[obs] <= upper[obs])

    df_filtered = df[mask].copy()

    labels = {obs: str(obs).replace("_", " ") for obs in obs_to_plot}
    unique_particles = [p for p in df_filtered["particle"].unique() if p in particle_colours]

    n_obs = len(obs_to_plot)
    # Increase width to make space for legend
    fig_width = size[0] + 2.5  # add space for legend
    fig_height = size[1]
    fig, axes = plt.subplots(
        n_obs, n_obs,
        figsize=(fig_width, fig_height),
        squeeze=False
    )

    # For each pair of observables, plot scatter or histogram
    for i, y_obs in enumerate(obs_to_plot):
        for j, x_obs in enumerate(obs_to_plot):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histogram
                for particle in unique_particles:
                    data = df_filtered[df_filtered["particle"] == particle][x_obs]
                    ax.hist(
                        data,
                        bins=50,
                        color=particle_colours[particle],
                        alpha=1,
                        label=particle if j == 0 else None,
                        histtype='stepfilled',
                        linewidth=1.2
                    )
            else:
                # Off-diagonal: scatter
                for particle in unique_particles:
                    mask = df_filtered["particle"] == particle
                    ax.scatter(
                        df_filtered.loc[mask, x_obs],
                        df_filtered.loc[mask, y_obs],
                        s=8,
                        color=particle_colours[particle],
                        alpha=0.9,
                        label=particle if (i == 0 and j == 1) else None,
                        edgecolors='none'
                    )
            # Axis labels
            if i == len(obs_to_plot) - 1:
                ax.set_xlabel(labels[x_obs], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[y_obs], fontsize=8)
            else:
                ax.set_yticklabels([])

    # Set the main title
    if title is not None:
        fig.suptitle(title, fontsize=14, y=1.02)

    # Create a single legend outside the grid (on the right)
    # Use the first diagonal axis to get handles/labels
    handles = []
    labels_legend = []
    for particle in unique_particles:
        # Create a dummy handle for each particle
        handles.append(Patch(color=particle_colours[particle], label=particle))
        labels_legend.append(particle)
    # Place the legend outside the right of the last column
    fig.legend(
        handles,
        labels_legend,
        title="Particle",
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])  # leave space for legend on right

def plot_feature_importances(model, x_train, book=None, plot=False, size=(12, 8)):
    """Plot the mean decrease in impurity (MDI) for each feature."""
    importances = model.feature_importances_
    importances = pd.Series(importances, index=x_train.columns)

    fig, ax = plt.subplots(figsize=size)
    try:
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        importances.plot.bar(yerr=std, ax=ax)
    except:
        print("Cannot calculate standard deviation of feature importances.")
        
    importances.plot.bar(ax=ax)

    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")

    if book is not None:
        book.Save()

    if not plot:
        plt.close()

def plot_confusion_matrix(cm, info, labels, book=None, plot=False, size=(20, 15)):
    """Plot confusion matrix."""
    plt.figure(figsize=size)
    sns.heatmap(cm, annot=info, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels[::-1], cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Final State Particle Selection Confusion Matrix', fontsize=16)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()

    if book is not None:
        book.Save()

    if not plot:
        plt.close()

def plot_roc_curve(binary_y_test, model_probs, class_label="$\\pi^{\\pm}$"):
    """
    Plot ROC curve for multiple models. 
    model_probs is a dictionary of model names and their probabilities for the given class.
    binary_y_test is the binary labels of the test set where 1 is the given class and 0 is any other class.
    """

    plt.figure(figsize=(15, 12))
    ax = plt.gca()

    # Set a nice color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    lw = 2.5

    # Plot ROC curves for each model
    for idx, (model, prob) in enumerate(model_probs.items()):
        color = colors[idx % len(colors)]
        disp = RocCurveDisplay.from_predictions(
            binary_y_test,
            prob,
            ax=ax,
            name=f"{model}",
            color=color,
            linestyle='-',
            linewidth=lw
        )

    # Plot the "No Skill" line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='No Skill', alpha=0.7, zorder=0)

    # Prettify axes
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=14, labelpad=10)
    ax.set_ylabel("True Positive Rate", fontsize=14, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle=':', linewidth=1, alpha=0.5)

    # Prettify legend
    legend = plt.legend(
        fontsize=12,
        loc='lower right',
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        borderpad=1,
        title="Model",
        title_fontsize=13
    )
    legend.get_frame().set_edgecolor('gray')

    # Prettify title
    plt.title(f"ROC Curve for {class_label} Classification", fontsize=16, pad=15, weight='bold')

    # Tight layout and show
    plt.tight_layout()
    plt.show()

# =============================================================================
# CUSTOM LOSS FUNCTIONS
# =============================================================================

def binary_purity_loss(y_pred, dtrain):
    """
    Custom loss function that optimizes for purity of correctly identified labels.
    """
    y_true = dtrain.get_label()
    
    # convert raw scores to probabilities using sigmoid
    prob = 1.0 / (1.0 + np.exp(-y_pred))
    
    # LOSS COMPONENTS
    
    # 1. Standard log loss component
    epsilon = 1e-15  # Small value to prevent log(0)
    prob = np.clip(prob, epsilon, 1 - epsilon)
    log_loss = -(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob))
    
    # 2. Confidence penalty: penalize predictions close to 0.5 (uncertain)
    confidence_penalty = 4 * (prob - 0.5) ** 2  # Quadratic penalty, max at 0.5
    
    # 3. Purity bonus: reward high confidence correct predictions
    correct_high_conf = np.where(
        ((y_true == 1) & (prob > 0.8)) | ((y_true == 0) & (prob < 0.2)),
        -0.5,  # Negative value = reward
        0  # No reward
    )
    
    # 4. Heavy penalty for confident wrong predictions
    wrong_high_conf = np.where(
        ((y_true == 0) & (prob > 0.8)) | ((y_true == 1) & (prob < 0.2)),
        2.0,  # Heavy penalty
        0
    )
    
    # Combine all components
    total_loss = log_loss + confidence_penalty + correct_high_conf + wrong_high_conf
    
    # GRADIENT CALCULATIONS

    # For gradient calculation, we need derivative with respect to raw predictions (before sigmoid)
    
    # Standard logistic gradient
    grad_log = prob - y_true
    
    # Confidence penalty gradient (derivative of 4*(prob - 0.5)^2 w.r.t. raw prediction)
    grad_conf = 8 * (prob - 0.5) * prob * (1 - prob)
    
    # High confidence correct prediction gradient
    grad_correct = np.where(
        ((y_true == 1) & (prob > 0.8)) | ((y_true == 0) & (prob < 0.2)),
        -0.5 * prob * (1 - prob),  # Derivative of reward term
        0
    )
    
    # High confidence wrong prediction gradient
    grad_wrong = np.where(
        ((y_true == 0) & (prob > 0.8)) | ((y_true == 1) & (prob < 0.2)),
        2.0 * prob * (1 - prob),  # Derivative of penalty term
        0
    )
    
    grad = grad_log + grad_conf + grad_correct + grad_wrong
    
    # HESSIAN CALCULATIONS

    # Standard logistic hessian
    hess_log = prob * (1 - prob)
    
    # Confidence penalty hessian
    hess_conf = 8 * prob * (1 - prob) * (1 - 2*prob + 2*prob*(prob - 0.5))
    
    # High confidence terms hessians (simplified)
    hess_correct = np.where(
        ((y_true == 1) & (prob > 0.8)) | ((y_true == 0) & (prob < 0.2)),
        -0.5 * prob * (1 - prob) * (1 - 2*prob),
        0
    )
    
    hess_wrong = np.where(
        ((y_true == 0) & (prob > 0.8)) | ((y_true == 1) & (prob < 0.2)),
        2.0 * prob * (1 - prob) * (1 - 2*prob),
        0
    )
    
    hess = hess_log + hess_conf + hess_correct + hess_wrong
    
    # Ensure hessian is positive for numerical stability
    hess = np.maximum(hess, 1e-8)
    
    return grad, hess


def multiclass_purity_loss(y_pred, dtrain):
    """
    Multi-class purity loss function that optimizes for purity of correctly identified labels.

    Purity components:
    1. Standard multi-class log loss (categorical cross-entropy)
    2. Confidence penalty for uncertain predictions (low max probability)
    3. Purity bonus for high confidence correct predictions
    4. Heavy penalty for confident wrong predictions
    
    y_pred: RAW prediction scores (shape: n_samples, n_classes)
    dtrain: Training data (DMatrix)
    """
    # get true labels and determine problem dimensions
    y_true = dtrain.get_label().astype(int)  # this has already been encoded to be integers
    n_samples = len(y_true)
    n_classes = y_pred.shape[1]
    
    # reshape y_pred to (n_samples, n_classes)
    #raw_scores = y_pred.reshape(n_samples, n_classes)

    # convert raw scores to probabilities (softmax)
    exp_scores = np.exp(y_pred)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # clip probabilities to prevent log(0)
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)
    
    # create one-hot encoded true labels
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true] = 1
    
    # LOSS COMPONENTS
    
    # 1. standard multi-class log loss (categorical cross-entropy)
    log_loss = -np.sum(y_onehot * np.log(probs), axis=1)  # this will be an array of shape (n_samples,) with the log loss for each sample
    
    # 2. confidence penalty: penalize uncertain predictions
    # penalize when max probability is low (uncertain predictions)
    max_prob = np.max(probs, axis=1)
    confidence_penalty = 2.0 * (0.8 - max_prob) ** 2 * (max_prob < 0.8)
    
    # 3. purity bonus: reward high confidence correct predictions
    correct_class_prob = probs[np.arange(n_samples), y_true]
    purity_bonus = np.where(
        correct_class_prob > 0.85,
        -0.8,  # Strong reward for very confident correct predictions
        np.where(
            correct_class_prob > 0.7,
            -0.3,  # Moderate reward for confident correct predictions
            0
        )
    )
    
    # 4. Heavy penalty for confident wrong predictions
    # Find cases where model is confident but wrong
    predicted_class = np.argmax(probs, axis=1)
    is_wrong = (predicted_class != y_true)
    is_confident_wrong = is_wrong & (max_prob > 0.8)
    confident_wrong_penalty = np.where(is_confident_wrong, 3.0, 0)
    
    # 5. Moderate penalty for somewhat confident wrong predictions
    moderate_wrong_penalty = np.where(
        is_wrong & (max_prob > 0.6) & (max_prob <= 0.8),
        1.5,
        0
    )
    
    # Combine all loss components (per sample)
    total_loss_per_sample = (log_loss + confidence_penalty + purity_bonus + 
                            confident_wrong_penalty + moderate_wrong_penalty)
    
    # ===========================================
    # GRADIENT CALCULATIONS
    # ===========================================
    
    # Initialize gradients (n_samples, n_classes)
    grad_matrix = np.zeros((n_samples, n_classes))
    
    # 1. Standard multi-class logistic gradient
    grad_log = probs - y_onehot
    
    # 2. Confidence penalty gradient
    # d/df_k [confidence_penalty] where confidence_penalty depends on max(softmax)
    grad_conf = np.zeros((n_samples, n_classes))
    max_class_idx = np.argmax(probs, axis=1)
    
    for i in range(n_samples):
        if max_prob[i] < 0.8:
            # Gradient of confidence penalty w.r.t. the class with max probability
            k_max = max_class_idx[i]
            coeff = 2.0 * 2.0 * (0.8 - max_prob[i]) * (-1)  # Chain rule
            
            # Gradient w.r.t. max class
            grad_conf[i, k_max] = coeff * probs[i, k_max] * (1 - probs[i, k_max])
            
            # Gradient w.r.t. other classes
            for k in range(n_classes):
                if k != k_max:
                    grad_conf[i, k] = coeff * (-probs[i, k_max] * probs[i, k])
    
    # 3. Purity bonus gradient (for correct class only)
    grad_purity = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        true_class = y_true[i]
        if correct_class_prob[i] > 0.85:
            coeff = -0.8
        elif correct_class_prob[i] > 0.7:
            coeff = -0.3
        else:
            coeff = 0
        
        if coeff != 0:
            # Gradient w.r.t. correct class
            grad_purity[i, true_class] = coeff * probs[i, true_class] * (1 - probs[i, true_class])
            # Gradient w.r.t. other classes
            for k in range(n_classes):
                if k != true_class:
                    grad_purity[i, k] = coeff * (-probs[i, true_class] * probs[i, k])
    
    # 4. Confident wrong penalty gradient
    grad_wrong = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        if is_confident_wrong[i]:
            pred_class = predicted_class[i]
            coeff = 3.0
            
            # Gradient w.r.t. predicted (wrong) class
            grad_wrong[i, pred_class] = coeff * probs[i, pred_class] * (1 - probs[i, pred_class])
            # Gradient w.r.t. other classes
            for k in range(n_classes):
                if k != pred_class:
                    grad_wrong[i, k] = coeff * (-probs[i, pred_class] * probs[i, k])
    
    # 5. Moderate wrong penalty gradient (similar to above but different coefficient)
    grad_moderate_wrong = np.zeros((n_samples, n_classes))
    moderate_wrong_mask = is_wrong & (max_prob > 0.6) & (max_prob <= 0.8)
    
    for i in range(n_samples):
        if moderate_wrong_mask[i]:
            pred_class = predicted_class[i]
            coeff = 1.5
            
            grad_moderate_wrong[i, pred_class] = coeff * probs[i, pred_class] * (1 - probs[i, pred_class])
            for k in range(n_classes):
                if k != pred_class:
                    grad_moderate_wrong[i, k] = coeff * (-probs[i, pred_class] * probs[i, k])
    
    # Combine all gradients
    grad_total = grad_log + grad_conf + grad_purity + grad_wrong + grad_moderate_wrong
    
    # ===========================================
    # HESSIAN CALCULATIONS (Simplified)
    # ===========================================
    
    # For multi-class, hessian is complex. We use simplified diagonal approximation
    hess_matrix = np.zeros((n_samples, n_classes))
    
    # Standard multi-class hessian (diagonal approximation)
    hess_log = probs * (1 - probs)
    
    # Additional hessian terms (simplified)
    hess_additional = 0.1 * probs * (1 - probs)  # Small additional curvature
    
    hess_matrix = hess_log + hess_additional
    
    # Ensure positive hessian for numerical stability
    hess_matrix = np.maximum(hess_matrix, 1e-6)
    
    # Flatten gradients and hessians to match XGBoost expected format
    grad_flat = grad_total.flatten()
    hess_flat = hess_matrix.flatten()
    
    return grad_flat, hess_flat