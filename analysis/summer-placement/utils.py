from re import I
import numpy as np
from particle import Particle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from python.analysis.Tags import GenerateTrueParticleTagsPiPlus
import os
import pickle
from datetime import datetime
import awkward as ak
import plotly.express as px
import re
from pandas.plotting import scatter_matrix
from matplotlib.ticker import MaxNLocator


def find_particle_from_tags(tags, event, track):
    for k, v in tags.items():
        if v.mask[event][track]:
            return k


def extract_observables(mc, size=-1, start=0, beam_selection_mask=None, verbose=False):
    """
    Extracts the observables from the ntuple.
    Returns a list of dictionaries, each containing the observables for a single track.
    size is the number of events to process. If -1, all events are processed.
    """

    if size == -1 or size > len(mc.recoParticles.track_chi2_proton):  # if size is not specified or larger than the number of events, process all events
        num_events = len(mc.recoParticles.track_chi2_proton)  # any of the flattened arrays can be used to get the number of events
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
                    "residual_range_mean": np.mean(mc.recoParticles.residual_range[event][track]),
                    "residual_range_median": np.median(mc.recoParticles.residual_range[event][track]),

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


                    # custom
                    #"hit_density": mc.recoParticles.n_hits_collection[event][track] / mc.recoParticles.track_len[event][track],
                    #"angle_to_beam": np.dot(np.array(list(mc.recoParticles.beam_startPos[event].to_list().values())) - np.array(list(mc.recoParticles.beam_endPos[event].to_list().values())), np.array(list(mc.recoParticles.track_start_dir[event][track].to_list().values()))),
                    #"distance_from_beam": find_distance(mc.recoParticles.track_start_pos[event][track], mc.recoParticles.beam_endPos[event])
                }

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


def find_distance(pos_1, pos_2):
    """
    Finds the position between two 3D position vectors.
    """
    return np.sqrt((pos_1["x"] - pos_2["x"])**2 + (pos_1["y"] - pos_2["y"])**2 + (pos_1["z"] - pos_2["z"])**2)


def clean_df(df, return_dropped=False, verbose=False):
    """
    Clean the dataframe by removing rows with missing entries.
    """
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
    """
    Convert the particle type to a binary classification.
    """
    for i in range(len(y)):
        if y[i] == "$\pi^{\pm}$":
            y[i] = "1"
        else:
            y[i] = "0"
    return y


def split_data(df, test_size=0.2, random_state=42, verbose=False, binary_classification=False):
    """
    Split the dataframe into training and testing sets.
    """
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


def accuracy(y_pred, y_test):
    """
    Accuracy is the ratio of correctly identified particles to the total number of particles.
    """
    return np.sum(y_pred == y_test) / len(y_test)


def count_items_in_list(list, X : list):
    """
    Counts the number of particles of a given type (or set of types) in a list.
    """
    count = 0
    for item in list:
        if item in X:
            count += 1
    return count


def classify_X_as_Y(list_a, list_b, X : list, Y : list):
    """
    Counts the number of times a particle of type X is identified as a particle of type Y. 
    Note X and Y are lists of particles, this allows for us to consider multiple particles at once e.g. pi± efficiency as opposed to just pi+ or pi- efficiency.
    """
    X_as_Y = 0
    for i in range(len(list_a)):
        if list_a[i] in X and list_b[i] in Y:
                X_as_Y += 1
    return X_as_Y


def purity(y_pred, y_test, identified_particles : list, true_particles : list):
    """
    Purity is the ratio of correctly identified particles to the total number of identified particles (of a given particle type(s)).
    """
    matched = classify_X_as_Y(y_pred, y_test, identified_particles, true_particles)
    total_identified = count_items_in_list(y_pred, identified_particles)
    if total_identified == 0:
        return 0
    else:
        return matched / total_identified


def efficiency(y_pred, y_test, identified_particles : list, true_particles : list, dropped_particles : list = None):
    """
    Efficiency is the ratio of correctly identified particles to the total number of real particles (of a given particle type(s)).
    """
    matched = classify_X_as_Y(y_pred, y_test, identified_particles, true_particles)
    total_real = count_items_in_list(y_test, true_particles)
    if dropped_particles is not None:
        total_real += count_items_in_list(dropped_particles, true_particles)
    if total_real == 0:
        return 0
    else:
        return matched / total_real


def get_pdg_id(name):
    """
    Returns the PDG ID for a given particle name.
    """
    return int(Particle.from_name(name).pdgid)


def get_pdg_ids(names):
    """
    Returns the PDG IDs for a list of particle names.
    """
    return [get_pdg_id(name) for name in names]


def plot_feature_importances(model, x_train):
    """
    Plots the mean decrease in impurity (MDI) for each feature. At each node in a decision tree (e.g. track_dEdX_mean > 2?), the decrease in impurity due to the feature in question is calculated. 
    The accumulated impurity decrease for each tree is calculated, and the mean of this is the MDI for that feature.
    """
    importances = model.feature_importances_
    importances = pd.Series(importances, index=x_train.columns)

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        importances.plot.bar(yerr=std, ax=ax)
    except:
        print("Cannot calculate standard deviation of feature importances.")
        
    importances.plot.bar(ax=ax)

    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    plt.show()


def create_confusion_matrix(y_test, y_pred, dropped_particles=None):
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


def plot_confusion_matrix(cm, info, labels,):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=info, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels[::-1], cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Final State Particle Selection Confusion Matrix', fontsize=16)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.show()


def master_pion_selection(data=None, ntuple=None, size=-1, model=None, custom_observables=None, verbose=False, plot_importances=False, plot_cm=False, plot_obs_1d=False, plot_obs_corr=False, plot_obs_scatter=False, binary_classification=False, drop_cols=None, consider_dropped_rows=False, return_performance=False):
    """
    Master function to run the ML selection. Simplest case is to provide just the ntuple (i.e. mc data) and the rest is done automatically.
    ntuple and size only need to be specified if data is not provided.
    Dropped particles are the particles that were dropped due to beam selection or other reasons. consider_dropped_rows=True just takes these into account when calculating efficiency.
    """
    # extract observables if not provided
    if data is None:
        if ntuple is None:
            raise ValueError("Either data or ntuple must be provided.")
        data = extract_observables(ntuple, size)

    # add custom observables if provided
    if custom_observables is not None:
        data = data | custom_observables
        
    df = pd.DataFrame(data)

    if consider_dropped_rows:
        df, dropped_rows = clean_df(df, return_dropped=True, verbose=verbose)
        dropped_particles = list(dropped_rows["particle"])
    else:
        df = clean_df(df, return_dropped=False, verbose=verbose)
        dropped_particles = None

    if drop_cols is not None:
        df.drop(columns=drop_cols, inplace=True)
        
    if binary_classification:
        x_train, x_test, y_train, y_test = split_data(df, verbose=verbose, binary_classification=True)
        if dropped_particles is not None:
            dropped_particles = convert_to_binary(dropped_particles)
    else:
        x_train, x_test, y_train, y_test = split_data(df, verbose=verbose, binary_classification=False)
    

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42);

    if verbose:
        print("Fitting model...")
    model.fit(x_train, y_train);
    if verbose:
        print("Model fitted.")

    y_pred = model.predict(x_test);
    y_pred = le.inverse_transform(y_pred)
    y_test = y_test.tolist();

    if binary_classification:
        pions = ["1"]
        not_pions = ["0"]
    else:
        pions = ["$\pi^{\pm}$"]

    pion_purity = purity(y_pred, y_test, pions, pions)
    pion_efficiency = efficiency(y_pred, y_test, pions, pions, dropped_particles)
    pion_count = count_items_in_list(y_test, pions)

    results_dict = {
        "Metric": [
            "Pion Count",
            "Total Test Particles",
            "Accuracy",
            "pi± Purity",
            "pi± Efficiency",
            "Purity x Efficiency"
        ],
        "Value": [
            pion_count,
            len(y_test),
            f"{accuracy(y_pred, y_test):.3f}",
            f"{pion_purity:.3f}",
            f"{pion_efficiency:.3f}",
            f"{pion_purity * pion_efficiency:.3f}"
        ]
    }
    results_df = pd.DataFrame(results_dict)

    if verbose:
        print(results_df)

    if plot_importances:
        plot_feature_importances(model, x_train)
    
    if plot_cm:
        cm, info, labels = create_confusion_matrix(y_test, y_pred, dropped_particles)
        plot_confusion_matrix(cm, info, labels)

    df_before_classification = x_test.copy()
    df_before_classification["particle"] = y_test

    mask = np.array(y_pred) == pions[0]
    df_after_classification = df_before_classification.iloc[mask]

    if plot_obs_1d:
        plot_observables_grid(df_after_classification, percentiles=[0.1, 0.9], norm=True)

    if plot_obs_corr:
        plot_correlation_matrix(df_after_classification, title="Correlation Matrix of Observables after classification", size=(10, 8))

    if plot_obs_scatter:
        plot_scatter_matrix(df_before_classification, title=f"Observables Scatter Matrix Before Classification ({len(df_after_classification)})", size=(50, 40), save=False)

    if return_performance:
        return model, pion_purity, pion_efficiency
    else:
        return model


def plot_tuning_results(hp_name, hp_values, pion_purities, pion_efficiencies):
    """
    Plots the purity and efficiency tuning for a given hyper parameter (hp).
    """
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
    plt.show()


def tune_hp(hp_tuning, data, model_name, plot=False, save=True, verbose=False, binary_classification=False, consider_dropped_rows=False):
    """
    Tunes a hyper parameter (hp) for a given model.
    """
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

            model, pion_purity, pion_efficiency = master_pion_selection(data=data, model=model, verbose=False, plot_importances=False, plot_cm=False, binary_classification=binary_classification, consider_dropped_rows=consider_dropped_rows, return_performance=True);
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
    output_dir = "/home/pemb6649/pi0-analysis/analysis/summer-placement/hp_tuning_results/"
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}{model_name}_{current_time}.pkl"

    with open(filename, "wb") as file:
        pickle.dump(hp_tuning_results, file)


def find_best_hps(hp_tuning_results):
    pass


def apply_track_operation(data, operation):
    """
    Apply an operation to each track in the data.
    """
    return ak.Array([[operation(track) for track in event] for event in data])


def plot_observables_grid(df, percentiles=[0.1, 0.9], ncols=4, figsize=(20, 15), norm=False):
    """
    Create a grid of plots for multiple observables using a pandas DataFrame.
    Assumes 'particle' column contains the tag/label for each row.
    """
    # Exclude the 'particle' column from observables to plot
    obs_names = [col for col in df.columns if col != "particle"]
    n_obs = len(obs_names)
    nrows = (n_obs + ncols - 1) // ncols  # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.flatten()

    # Get unique particle tags in the order they appear
    tags = list(df["particle"].unique())

    for i, obs_name in enumerate(obs_names):
        ax = axes_flat[i]

        # Split data by tags
        split_data = [df.loc[df["particle"] == tag, obs_name] for tag in tags]

        # Concatenate all data for x-range calculation
        all_data = np.concatenate([d for d in split_data])

        lower = np.percentile(all_data, percentiles[0]*100)
        upper = np.percentile(all_data, percentiles[1]*100)
        x_range = (lower, upper)

        # Plot stacked histogram
        ax.hist(
            split_data,
            bins=100,
            range=x_range,
            stacked=True,
            label=[str(tag).replace("_", " ") for tag in tags],
            density=norm
        )

        ax.set_xlabel(obs_name.replace('_', ' '))
        ax.set_ylabel('Normalized entries' if norm else 'Entries')
        ax.set_title(obs_name.replace('_', ' '))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_obs, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()


def plot_correlation_matrix(df, obs_to_plot=None, size=(10, 8), title=None):
    """
    Plot a correlation matrix of the observables using seaborn.
    """

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


def plot_scatter_matrix(df, obs_to_plot=None, title=None, size=(10, 8), save=False):
    """
    Plot a scatter matrix of the observables using matplotlib.
    Only include data within the 10th and 90th percentiles for each observable.
    Uses a consistent colour scheme for the particles.
    The legend is placed outside the plots (on the right).

    THIS FUNCTION ORIGINALLY USED PLOTLY BUT WAS REWRITTEN TO USE MATPLOTLIB WITH AI - SO THE CODE MAY BE A BIT ODD.
    """
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
        from matplotlib.patches import Patch
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
    if save:
        filename = "/home/pemb6649/pi0-analysis/analysis/summer-placement/observables-visualisation/scatter_matrix.png"
        try:
            plt.savefig(filename, bbox_inches='tight', dpi=200)
            print(f"Plot saved to {filename}")
        except Exception as e:
            print("Could not save plot as PNG.")
            print(f"Error: {e}")
    else:
        plt.show()
