# Imports
import sys
sys.path.insert(1, '/users/wx21978/projects/pion-phys/pi0-analysis/analysis/')
import os
from importlib import reload
import copy
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.models import gat_v2
from tensorflow import keras
import sklearn.ensemble as ensemble
import sklearn.impute as impute
import awkward as ak
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colours
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# from python.analysis import EventSelection, Plots, vector, PairSelection, Master, PFOSelection, cross_section, CutOptimization
from python.analysis import Plots
from python.analysis import SelectionEvaluation as seval
from iminuit import cost, Minuit
import scipy.stats as scistats
from python.gnn import DataPreparation, Models, bdt_classifier, Layers, Fitter
# import apps.cex_analysis_input as cai
from apps import photon_pairs
from scipy.stats import poisson, rv_histogram
import scipy.optimize as opt
import time
import timeit

out_folder = "/users/wx21978/projects/pion-phys/plots/robustness/"

plt_conf = Plots.PlotConfig()
plt_conf.SHOW_PLOT = False
plt_conf.SAVE_FOLDER = out_folder
# plt_conf.BINS = 30

model_path = "/users/wx21978/projects/pion-phys/analyses/3GeV_MC_only/gnn_data/models/Model_data_PNA_0"

template_path_params = [
    DataPreparation.create_filepath_dictionary("/users/wx21978/projects/pion-phys/analyses/3GeV_both/gnn_data/3GeV_MC_Set01_final_07-08-24"),
    DataPreparation.create_filepath_dictionary("/users/wx21978/projects/pion-phys/analyses/3GeV_both/gnn_data/3GeV_MC_Set02_final_07-08-24")]

loaded_model_mc = Models.load_model_from_file(model_path, new_data_folder=template_path_params[0]["folder_path"])

temp_pred, temp_truth = Models.get_predictions(
    loaded_model_mc, template_path_params[0]["schema_path"], template_path_params[0]["test_path"])
mc_data_pred, mc_data_truth = Models.get_predictions(
    loaded_model_mc, template_path_params[1]["schema_path"], template_path_params[1]["test_path"])
