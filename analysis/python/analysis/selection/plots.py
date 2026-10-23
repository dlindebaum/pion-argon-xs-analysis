"""Selection plotting helpers."""
import os

import awkward as ak
import numpy as np
import pandas as pd

from python.analysis import plots

x_label = {
    "track_score_all" : "Track score", 
    "TrackScoreCut" : "Track score",
    "NHitsCut" : "Number of hits",
    "PiPlusSelection" : "Median $dE/dX$ (MeV/cm)",
    "BeamParticleDistanceCut" : "$d$ (cm)",
    "BeamParticleIPCut" : "$b$ (cm)",
    "Chi2ProtonSelection" : "$(\\chi^{2}/ndf)_{p}$",
    "PiBeamSelection" : "True particle ID",
    "APA3Cut" : "Beam end position $z$ (cm)",
    "TrueFiducialCut" : "True beam end position $z$ (cm)",
    "PandoraTagCut" : "Pandora tag",
    "DxyCut" : "$\\delta_{xy}$",
    "DzCut" : "$\\delta_{z}$",
    "CosThetaCut" : "$\\cos(\\theta)$",
    "MichelScoreCut" : "Michel score",
    "MedianDEdXCut" : "Median $dE/dX$ (MeV/cm)",
    "BeamScraperCut" : "$r_{inst}$",
    "TrackLengthSelection" : "l (cm)"
}
y_scale = {
    "track_score_all" : "log",
    "TrackScoreCut" : "linear",
    "NHitsCut" : "linear",
    "PiPlusSelection" : "log",
    "BeamParticleDistanceCut" : "linear",
    "BeamParticleIPCut" : "linear",
    "Chi2ProtonSelection" : "linear",
    "PiBeamSelection" : None,
    "APA3Cut" : "log",
    "TrueFiducialCut" : "linear",
    "PandoraTagCut" : None,
    "DxyCut" : "log",
    "DzCut" : "log",
    "CosThetaCut" : "log",
    "MichelScoreCut" : "log",
    "MedianDEdXCut" : "log",
    "BeamScraperCut" : "linear",
    "TrackLengthSelection" : "linear"
}
x_range = {
    "track_score_all" : None,
    "TrackScoreCut" : [0, 1],
    "NHitsCut" : [0, 500],
    "PiPlusSelection" : [0, 10],
    "BeamParticleDistanceCut" : [0, 150],
    "BeamParticleIPCut" : [0, 100],
    "Chi2ProtonSelection" : [0, 600],
    "PiBeamSelection" : None,
    "PandoraTagCut" : None,
    "APA3Cut" : [0, 700],
    "TrueFiducialCut" : [0, 700],
    "DxyCut" : [0, 5],
    "DzCut" : [-5, 5],
    "CosThetaCut" : [0.9, 1],
    "MichelScoreCut" : [0, 1],
    "MedianDEdXCut" : [1.5, 3],
    "BeamScraperCut" : [0, 5],
    "TrackLengthSelection" : [0, 400],

}
nbins = {
    "track_score_all" : 50,
    "TrackScoreCut" : 50,
    "NHitsCut" : 50,
    "PiPlusSelection" : 50,
    "BeamParticleDistanceCut" : 50,
    "BeamParticleIPCut" : 50,
    "Chi2ProtonSelection" : 50,
    "PiBeamSelection" : 50,
    "APA3Cut" : np.append(np.linspace(0, 220, 11), 700),
    "TrueFiducialCut" : 50,
    "PandoraTagCut" : 50,
    "DxyCut" : 50,
    "DzCut" : 50,
    "CosThetaCut" : 50,
    "MichelScoreCut" : 50,
    "MedianDEdXCut" : 50,
    "BeamScraperCut" : 50,
    "TrackLengthSelection" : 50
}
ncols = {
    "track_score_all" : 2,
    "TrackScoreCut" : 2,
    "NHitsCut" : 2,
    "PiPlusSelection" : 2,
    "BeamParticleDistanceCut" : 2,
    "BeamParticleIPCut" : 2,
    "Chi2ProtonSelection" : 2,
    "PiBeamSelection" : 2,
    "APA3Cut" : 2,
    "TrueFiducialCut" : 2,
    "PandoraTagCut" : 2,
    "DxyCut" : 2,
    "DzCut" : 2,
    "CosThetaCut" : 2,
    "MichelScoreCut" : 2,
    "MedianDEdXCut" : 2,
    "BeamScraperCut" : 2,
    "TrackLengthSelection" : 2
}
truncate = {
    "track_score_all" : True,
    "TrackScoreCut" : True,
    "NHitsCut" : False,
    "PiPlusSelection" : False,
    "BeamParticleDistanceCut" : True,
    "BeamParticleIPCut" : True,
    "Chi2ProtonSelection" : True,
    "PiBeamSelection" : False,
    "APA3Cut" : False,
    "TrueFiducialCut" : False,
    "PandoraTagCut" : False,
    "DxyCut" : False,
    "DzCut" : False,
    "CosThetaCut" : False,
    "MichelScoreCut" : False,
    "MedianDEdXCut" : False,
    "BeamScraperCut" : False,
    "BeamScraperCut" : False,
    "TrackLengthSelection" : False
}


def CalculateArrowLength(value, x_range):
    if x_range is None:
        l = ak.max(value)
    else:
        l = max(x_range)
    return 0.1 * l


def MakeBeamSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float, book_name = "beam"):
    """ Beam particle selection plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
        norm (float): plot normalisation
    """
    norm = False if output_data is None else norm
    with plots.PlotBook(outDir + f"{book_name}.pdf") as pdf:

        for p in ["PiBeamSelection_endz", "no_selection_endz", "selection_endz"]:
            if p in output_mc:
                plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = "linear", x_label = x_label["APA3Cut"], bins = nbins["APA3Cut"], ncols = ncols["APA3Cut"], x_range = [30, 230], truncate = truncate["APA3Cut"])
                # plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                pdf.Save()

        for p in output_mc:
            if p in x_label:
                if p == "PiBeamSelection":
                    plots.PlotTags(output_mc["PiBeamSelection"]["tags"], "True particle ID")
                    pdf.Save()
                elif p == "PandoraTagCut":
                    if output_data is None:
                        plots.PlotBar(output_mc[p]["value"], xlabel = x_label[p])
                    else:
                        plots.PlotBarComparision(output_mc[p]["value"], output_data[p]["value"], label_1 = "MC", label_2 = "Data", xlabel = x_label[p], fraction = True, barlabel = True)
                    pdf.Save()

                elif p == "APA3Cut":
                    for l in ["linear", "log"]:
                        plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = l, x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                elif p == "TrueFiducialCut":
                    for l in ["linear", "log"]:
                        plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = None, norm = norm, y_scale = l, x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                elif p == "BeamScraperCut":
                    for t in ["BeamScraperCut", "BeamScraperCut_scraper_tag"]:
                        plots.PlotTagged(output_mc[p]["value"], output_mc[t]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                else:
                    plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                    if p == "CosThetaCut":
                        arrow_l = 0.02
                    else:
                        arrow_l = CalculateArrowLength(output_mc[p]["value"], x_range[p])
                    plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = arrow_l, arrow_loc = 0.7, color = "k")
                    pdf.Save()

        plots.PlotTags(output_mc["final_tags"]["tags"], "True particle ID")
        pdf.Save()
    plots.plt.close("all")
    return


def MakePFOSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float, book_name : str):
    norm = False if output_data is None else norm

    with plots.PlotBook(outDir + book_name) as pdf:

        for p in output_mc:
            if p in x_label:
                plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.5, color = "C6")
                plots.plt.ylim(bottom = 0)
                pdf.Save()
                if f"{p}_completeness" in output_mc:
                    plots.PlotHist2DImshowMarginal(ak.ravel(output_mc[p]["value"]), ak.ravel(output_mc[f"{p}_completeness"]["value"]), ylabel = "Completeness", xlabel = x_label[p], x_range = x_range[p], bins = nbins[p], norm = "column", c_scale = "linear")
                    plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.1, color = "C6")
                    pdf.Save()
        plots.PlotTags(output_mc["final_tags"]["tags"], xlabel = "True particle ID")
        pdf.Save()
    plots.plt.close("all")
    return


def MakePFOSelectionPlotsConsdensed(output_mc : dict, output_mc_loose : dict, output_data : dict, output_data_loose : dict, outDir : str, norm : float, book_name : str):
    norm = False if output_data is None else norm

    with plots.PlotBook(outDir + book_name) as pdf:

        for p in output_mc_loose:
            if p in x_label:
                plots.PlotTagged(output_mc_loose[p]["value"], output_mc_loose[p]["tags"], data2 = output_data_loose[p]["value"] if output_data_loose else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                
                for c, mc in zip(["C6", "magenta"], [output_mc, output_mc_loose]):
                    if p == "PiPlusSelection":
                        sf = 0.5
                    else:
                        sf = 1
                    plots.DrawMultiCutPosition(mc[p]["cuts"], face = mc[p]["op"], arrow_length = sf * CalculateArrowLength(mc[p]["value"], x_range[p]), arrow_loc = 0.5, color = c)
                plots.plt.ylim(bottom = 0)                
                pdf.Save()
                if f"{p}_completeness" in output_mc:
                    plots.PlotHist2DImshowMarginal(ak.ravel(mc[p]["value"]), ak.ravel(mc[f"{p}_completeness"]["value"]), ylabel = "Completeness", xlabel = x_label[p], x_range = x_range[p], bins = nbins[p], norm = "column", c_scale = "linear")

                    for c, mc in zip(["C6", "magenta"], [output_mc, output_mc_loose]):
                        plots.DrawMultiCutPosition(mc[p]["cuts"], face = mc[p]["op"], arrow_length = CalculateArrowLength(mc[p]["value"], x_range[p]), arrow_loc = 0.1, color = c)

                    pdf.Save()

        for mc in [output_mc, output_mc_loose]:
            plots.PlotTags(mc["final_tags"]["tags"], xlabel = "True particle ID")
            pdf.Save()
    plots.plt.close("all")
    return


def MakePi0SelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float, nbins : int):
    """ Pi0 selection plots.

    Args:
        output_mc (dict): mc to plot
        output_data (dict): data to plot
        outDir (str): output directory
        norm (float): plot normalisation
    """
    norm = False if output_data is None else norm

    with plots.PlotBook(outDir + "pi0.pdf") as pdf:
        if "NPhotonCandidateSelection" in output_mc:
            if output_data is not None:
                scale = ak.count(output_data["NPhotonCandidateSelection"]["value"]) / ak.count(output_mc["NPhotonCandidateSelection"]["value"])

                n_photons_scaled = []
                u, c = np.unique(output_mc["NPhotonCandidateSelection"]["value"], return_counts = True)
                for i, j in zip(u, c):
                    n_photons_scaled.extend([i]* int(scale * j))

                plots.PlotBarComparision(n_photons_scaled, output_data["NPhotonCandidateSelection"]["value"], xlabel = "Number of $\\pi^{0}$ photon candidates", label_1 = "MC", label_2 = "Data", fraction = True, barlabel = False)
            else:
                plots.PlotBar(output_mc["NPhotonCandidateSelection"]["value"], xlabel = "Number of $\\pi^{0}$ photon candidates")
            pdf.Save()

        if "Pi0MassSelection" in output_mc:
            plots.PlotTagged(output_mc["Pi0MassSelection"]["value"], output_mc["Pi0MassSelection"]["tags"], data2 = output_data["Pi0MassSelection"]["value"] if output_data else None, bins = nbins, x_label = "$m_{\\gamma\\gamma}$ (MeV)", x_range = [0, 500], norm = norm, ncols = 1)
            plots.DrawMultiCutPosition(output_mc["Pi0MassSelection"]["cuts"], face = output_mc["Pi0MassSelection"]["op"], arrow_length = 50, color = "C6")
            pdf.Save()

            plots.PlotTagged(output_mc["Pi0MassSelection"]["value"], output_mc["mass_event_tag"]["tags"], data2 = output_data["Pi0MassSelection"]["value"] if output_data else None, bins = nbins, x_label = "$m_{\\gamma\\gamma}$ (MeV)", x_range = [0, 500], norm = norm, ncols = 1)
            plots.DrawMultiCutPosition(output_mc["Pi0MassSelection"]["cuts"], face = output_mc["Pi0MassSelection"]["op"], arrow_length = 50, color = "C6")
            pdf.Save()

        if "Pi0OpeningAngleSelection" in output_mc:
            plots.PlotTagged(output_mc["Pi0OpeningAngleSelection"]["value"], output_mc["Pi0OpeningAngleSelection"]["tags"], data2 = output_data["Pi0OpeningAngleSelection"]["value"] if output_data else None, bins = nbins, x_label = "$\\phi$ (rad)", norm = norm, ncols = 1)

            plots.DrawMultiCutPosition((np.array(output_mc["Pi0OpeningAngleSelection"]["cuts"]) * np.pi / 180).tolist(), face = output_mc["Pi0OpeningAngleSelection"]["op"], arrow_length = 0.25, color = "C6")
            plots.plt.ylim(bottom = 0)
            pdf.Save()
    plots.plt.close("all")
    return


def MakeRegionPlots(outputs_mc_masks : dict, outputs_data_masks : dict, outDir : str):
    """ Correlation matrices for truth and reco region selection.

    Args:
        outputs_mc_masks (dict): mc masks for each region
        outputs_data_masks (dict): data masks for each region
        outDir (str): output directory
    """
    with plots.PlotBook(outDir + "regions.pdf") as pdf:
        # Visualise the regions
        plots.plot_region_data(outputs_mc_masks["truth_regions"], compare_max=0, title="truth regions")
        pdf.Save()
        plots.plot_region_data(outputs_mc_masks["reco_regions"], compare_max=0, title="reco regions")
        pdf.Save()
        # Compare the regions
        plots.compare_truth_reco_regions(outputs_mc_masks["reco_regions"], outputs_mc_masks["truth_regions"], title="")
        pdf.Save()

        if outputs_data_masks is not None:
            plots.plot_region_data(outputs_data_masks["reco_regions"], compare_max=0, title="reco regions")
            pdf.Save()
    plots.plt.close("all")
    return

def MakeTables(output : dict, out : str, sample : str):
    """ Create cutflow tables.

    Args:
        output (dict): output data
        out (str): output name
        sample (str): sample name i.e. mc or data
    """
    for s in output:
        print(f"{s=}")
        if "table" in output[s]:
            outdir = out + s + "/"
            os.makedirs(outdir, exist_ok = True)
            df = output[s]["table"]
            df = df.rename(columns = {i : i.replace(" remaining", "") for i in df})
            print(df)
            purity = pd.concat([df["Name"], df.iloc[:, df.columns.to_list().index("Name") + 1:].div(df.iloc[:, df.columns.to_list().index("Name") + 1], axis = 0)], axis = 1)
            efficiency = pd.concat([df["Name"], df.iloc[:, df.columns.to_list().index("Name") + 1:].div(df.iloc[0, df.columns.to_list().index("Name") + 1:], axis = 1)], axis = 1)

            df.to_hdf(outdir + s + "_counts.hdf5", key='df')
            purity.to_hdf(outdir + s + "_purity.hdf5", key='df')
            efficiency.to_hdf(outdir + s + "_efficiency.hdf5", key='df')
            
            df.style.hide(axis = "index").to_latex(outdir + s + "_counts.tex")
            purity.style.hide(axis = "index").to_latex(outdir + s + "_purity.tex")
            efficiency.style.hide(axis = "index").to_latex(outdir + s + "_efficiency.tex")
    return
