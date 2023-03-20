#!/usr/bin/env python

# Imports
import sys
sys.path.insert(1, '/users/wx21978/projects/pion-phys/pi0-analysis/analysis/')
from python.analysis import (
    Master, vector, Plots, EventSelection, PairSelection)
import os
import numpy as np
import awkward as ak


#######################################################################
#######################################################################
##########                  PAIR PROPERTIES                  ##########
#######################################################################
#######################################################################

# def paired_mass(events, pair_coords):
#     """
#     Finds the mass of the pairs given by `pair_coords` from `events`
#     assuming relativistic limit.

#     Parameters
#     ----------
#     events : Data
#         Events from which the pairs are drawn.
#     pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
#         Inidicies to construct the pairs.

#     Returns
#     -------
#     mass : ak.Array
#         Invarient masses of the pairs.
#     """
#     # Get the momenta via the pair indicies
#     first_mom = events.recoParticles.momentum[pair_coords["0"]]
#     second_mom = events.recoParticles.momentum[pair_coords["1"]]
#     # Calculate
#     e = vector.magnitude(first_mom) + vector.magnitude(second_mom)
#     p = vector.magnitude(vector.add(first_mom, second_mom))
#     return np.sqrt(e**2 - p**2)


# def paired_momentum(events, pair_coords):
#     """
#     Finds the summed momenta of the pairs given by `pair_coords` from
#     `events`.

#     Parameters
#     ----------
#     events : Data
#         Events from which the pairs are drawn.
#     pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
#         Inidicies to construct the pairs.

#     Returns
#     -------
#     mom : ak.Array
#         Summed momenta of the pairs.
#     """
#     # Get the momenta via the pair indicies
#     first_mom = events.recoParticles.momentum[pair_coords["0"]]
#     second_mom = events.recoParticles.momentum[pair_coords["1"]]
#     # Calculate
#     return vector.add(first_mom, second_mom)


# def paired_energy(events, pair_coords):
#     """
#     Finds the summed energies of the pairs given by `pair_coords` from
#     `events`.

#     Parameters
#     ----------
#     events : Data
#         Events from which the pairs are drawn.
#     pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
#         Inidicies to construct the pairs.

#     Returns
#     -------
#     energy : ak.Array
#         Summed energies of the pairs.
#     """
#     # Get the energies via the pair indicies
#     first_mom = events.recoParticles.momentum[pair_coords["0"]]
#     second_mom = events.recoParticles.momentum[pair_coords["1"]]
#     # Calculate
#     return vector.magnitude(first_mom) + vector.magnitude(second_mom)


# def paired_closest_approach(events, pair_coords):
#     """
#     Finds the closest approaches of the pairs given by `pair_coords`
#     from `events`.

#     Parameters
#     ----------
#     events : Data
#         Events from which the pairs are drawn.
#     pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
#         Inidicies to construct the pairs.

#     Returns
#     -------
#     closest_approach : ak.Array
#         Distance of closest approach of the pairs.
#     """
#     first_dir = events.recoParticles.direction[pair_coords["0"]]
#     first_pos = events.recoParticles.startPos[pair_coords["0"]]
#     second_dir = events.recoParticles.direction[pair_coords["1"]]
#     second_pos = events.recoParticles.startPos[pair_coords["1"]]
#     return pfoProperties.closest_approach(
#         first_dir, second_dir, first_pos, second_pos)


# def paired_beam_impact(events, pair_coords):
#     """
#     Finds the impact parameter between the beam and the combined
#     momentum traced back from the midpoint of the closest approach of
#     each pair in `pair_coords` for each event in `events`.

#     Parameters
#     ----------
#     events : Data
#         Events from which the pairs are drawn.
#     pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
#         Inidicies to construct the pairs.

#     Returns
#     -------
#     impact_parameter : ak.Array
#         Impact parameter between the beam vertex and each pairs.
#     """
#     first_mom = events.recoParticles.momentum[pair_coords["0"]]
#     first_pos = events.recoParticles.startPos[pair_coords["0"]]
#     second_mom = events.recoParticles.momentum[pair_coords["1"]]
#     second_pos = events.recoParticles.startPos[pair_coords["1"]]
#     # Direction of the summed momenta of the PFOs
#     paired_direction = vector.normalize(vector.add(first_mom, second_mom))
#     # Midpoint of the line of closest approach between the PFOs
#     shared_vertex = pfoProperties.get_shared_vertex(
#         first_mom, second_mom, first_pos, second_pos)
#     # Impact parameter between the PFOs and corresponding beam vertex
#     return pfoProperties.get_impact_parameter(
#         paired_direction, shared_vertex, events.recoParticles.beam_endPos)


# def paired_separation(events, pair_coords):
#     """
#     Finds the separations between start positions of the pairs given
#     by `pair_coords` from `events`.

#     Parameters
#     ----------
#     events : Data
#         Events from which the pairs are drawn.
#     pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
#         Inidicies to construct the pairs.

#     Returns
#     -------
#     separation : ak.Array
#         Separation between start points of the pairs.
#     """
#     # Get the positions via the pair indicies
#     first_pos = events.recoParticles.startPos[pair_coords["0"]]
#     second_pos = events.recoParticles.startPos[pair_coords["1"]]
#     return pfoProperties.get_separation(first_pos, second_pos)


# def paired_beam_slice(events, pair_coords):
#     """
#     Finds the detector slice that both PFOs in the the pairs given
#     by `pair_coords` from `events`. If they do not have a shared
#     slice, returns -1 for the pair.

#     Parameters
#     ----------
#     events : Data
#         Events from which the pairs are drawn.
#     pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
#         Inidicies to construct the pairs.

#     Returns
#     -------
#     slice : ak.Array
#         Slice shared by the PFOs in the pairs.
#     """
#     # Get the positions via the pair indicies
#     first_slice = events.recoParticles.sliceID[pair_coords["0"]]
#     second_slice = events.recoParticles.sliceID[pair_coords["1"]]
#     return (first_slice == second_slice) * (first_slice + 1) - 1


# # N.B. this is phi in Shyam's scheme
# def paired_opening_angle(events, pair_coords):
#     """
#     Finds the opening angles of the pairs given by `pair_coords` from
#     `events`.

#     Parameters
#     ----------
#     events : Data
#         Events from which the pairs are drawn.
#     pair_coords : ak.zip({'0':ak.Array, '1':ak.Array})
#         Inidicies to construct the pairs.

#     Returns
#     -------
#     angle : ak.Array
#         Opening angle of the pairs.
#     """
#     # Get the momenta via the pair indicies
#     first_dir = events.recoParticles.direction[pair_coords["0"]]
#     second_dir = events.recoParticles.direction[pair_coords["1"]]
#     # Calculate
#     return np.arccos(vector.dot(first_dir, second_dir))


if __name__ == "__main__":

    print("Running pair analysis")

    plot_config = Plots.PlotConfig()
    plot_config.SAVE_FOLDER = ("/users/wx21978/projects/pion-phys/plots/"
                               + "new_ShowerPair_test")

    # Setting up the batch plotters:
    mass_plotter = Plots.PairHistsBatchPlotter(
        "mass", "MeV",
        plot_config=plot_config, range=[None, 1000, 400], bins=[500, 100, 40])
    momentum_plotter = Plots.PairHistsBatchPlotter(
        "momentum", "MeV",
        plot_config=plot_config, range=[None, 2000, 500], bins=[500, 200, 60])
    energy_plotter = Plots.PairHistsBatchPlotter(
        "energy", "MeV",
        plot_config=plot_config, range=[None, 2000, 500], bins=[500, 200, 60])
    approach_plotter = Plots.PairHistsBatchPlotter(
        "closest approach", "cm",
        plot_config=plot_config, range=[None, (-100, 50)], bins=[100, 50])
    separations_plotter = Plots.PairHistsBatchPlotter(
        "pfo separation", "cm",
        plot_config=plot_config, range=[None, 200], bins=[100, 30])
    impact_plotter = Plots.PairHistsBatchPlotter(
        "beam impact", "cm",
        plot_config=plot_config, range=[None, 200], bins=[100, 30])
    angles_plotter = Plots.PairHistsBatchPlotter(
        "angle", "rad",
        plot_config=plot_config, range=None, bins=100)

    # Make the bin widths that keep equal area on a sphere
    # Area cover over angle dTheta is r sin(Theta) dTheta (with r=1)
    # So we need constant sin(Theta) dTheta
    # In the range Theta = [0, pi), we have
    # \int^\pi_0 sin(\theta) d\theta = 2
    # So for 100 bins, we need: sin(Theta) dTheta = 2/100 = 0.02
    # \int^{\theta_new}_{\theta_old} sin(\theta) d\theta = 2/100
    # So 0.2 = cons(theta_old) - cos(theta_new)
    n_bins = 100
    sphere_bins = np.zeros(n_bins+1)
    for i in range(n_bins):
        sphere_bins[i + 1] = np.arccos(
            np.max([np.cos(sphere_bins[i]) - 2/n_bins, -1]))

    angles_sphere_plotter = Plots.PairHistsBatchPlotter(
        "angle", "arcrad",
        plot_config=plot_config,
        range=None, bins=sphere_bins,
        unique_save_id="_sphere", inc_norm=False)
    angles_sphere_norm_plotter = Plots.PairHistsBatchPlotter(
        "angle", "arcrad",
        plot_config=plot_config,
        range=None, bins=sphere_bins,
        unique_save_id="_sphere_norm", inc_norm=False)

    # Get the batches
    batch_folder = "/scratch/wx21978/pi0/root_files/1GeV_beam_v4_prelim/"

    batch_names = os.listdir(batch_folder)

    for batch in batch_names:
        print("Beginning batch: " + batch)

        evts = EventSelection.load_and_cut_data(
            batch_folder + batch,
            batch_size=-1, batch_start=-1,
            cnn_cut=0.5,
            n_hits_cut=80,
            beam_slice_cut=False,
            distance_bounds_cm=(3, 90),
            max_impact_cm=20)

        # truth_pair_indicies, valid_events = get_best_pairs(
        #     evts, method="mom", return_type="mask", report=True)

        # evts.Filter([valid_events], [valid_events])
        # truth_pair_indicies = truth_pair_indicies[valid_events]
        # del valid_events

        pair_coords = ak.argcombinations(evts.recoParticles.number, 2)

        # sig_count = pair_apply_sig_mask(truth_pair_indicies, pair_coords)
        # del truth_pair_indicies
        sig_count = PairSelection.get_sig_count(evts, pair_coords)

        pairs = Master.ShowerPairs(evts, pair_coords)

        print("Plotting masses...")
        mass_plotter.add_batch(
            pairs.reco_mass, sig_count)

        print("Plotting momenta...")
        momentum_plotter.add_batch(vector.magnitude(
            pairs.reco_pi0_mom), sig_count)

        print("Plotting energies...")
        energy_plotter.add_batch(
            pairs.reco_energy, sig_count)

        print("Plotting closest approaches...")
        approach_plotter.add_batch(
            pairs.reco_closest_approach,
            sig_count)

        print("Plotting separations...")
        separations_plotter.add_batch(
            pairs.reco_separation, sig_count)

        print("Plotting beam impact parameters...")
        impact_plotter.add_batch(
            PairSelection.paired_beam_impact(pairs), sig_count)

        print("Plotting opening angles...")
        angles = pairs.reco_angle
        angles_plotter.add_batch(angles, sig_count)

        print("Plotting opening angles in bins of equal arcradians...")
        angles_sphere_plotter.add_batch(angles, sig_count)
        angles_sphere_norm_plotter.add_batch(
            angles, sig_count, weights=ak.full_like(angles, 1/(0.04*np.pi)))
        del angles

    print("Making plots...")
    mass_plotter.make_figures()
    momentum_plotter.make_figures()
    energy_plotter.make_figures()
    approach_plotter.make_figures()
    separations_plotter.make_figures()
    impact_plotter.make_figures()
    angles_plotter.make_figures()
    angles_sphere_plotter.make_figures()
    angles_sphere_norm_plotter.make_figures()

    print("All plots made.")
