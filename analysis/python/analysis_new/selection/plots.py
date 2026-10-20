"""Selection plotting helpers."""

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
    with Plots.PlotBook(outDir + f"{book_name}.pdf") as pdf:

        for p in ["PiBeamSelection_endz", "no_selection_endz", "selection_endz"]:
            if p in output_mc:
                Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = "linear", x_label = x_label["APA3Cut"], bins = nbins["APA3Cut"], ncols = ncols["APA3Cut"], x_range = [30, 230], truncate = truncate["APA3Cut"])
                # Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                pdf.Save()

        for p in output_mc:
            if p in x_label:
                if p == "PiBeamSelection":
                    Plots.PlotTags(output_mc["PiBeamSelection"]["tags"], "True particle ID")
                    pdf.Save()
                elif p == "PandoraTagCut":
                    if output_data is None:
                        Plots.PlotBar(output_mc[p]["value"], xlabel = x_label[p])
                    else:
                        Plots.PlotBarComparision(output_mc[p]["value"], output_data[p]["value"], label_1 = "MC", label_2 = "Data", xlabel = x_label[p], fraction = True, barlabel = True)
                    pdf.Save()

                elif p == "APA3Cut":
                    for l in ["linear", "log"]:
                        Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = l, x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                elif p == "TrueFiducialCut":
                    for l in ["linear", "log"]:
                        Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = None, norm = norm, y_scale = l, x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                elif p == "BeamScraperCut":
                    for t in ["BeamScraperCut", "BeamScraperCut_scraper_tag"]:
                        Plots.PlotTagged(output_mc[p]["value"], output_mc[t]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                        Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.7, color = "k")
                        pdf.Save()
                else:
                    Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                    if p == "CosThetaCut":
                        arrow_l = 0.02
                    else:
                        arrow_l = CalculateArrowLength(output_mc[p]["value"], x_range[p])
                    Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = arrow_l, arrow_loc = 0.7, color = "k")
                    pdf.Save()

        Plots.PlotTags(output_mc["final_tags"]["tags"], "True particle ID")
        pdf.Save()
    Plots.plt.close("all")
    return


def MakePFOSelectionPlots(output_mc : dict, output_data : dict, outDir : str, norm : float, book_name : str):
    norm = False if output_data is None else norm

    with Plots.PlotBook(outDir + book_name) as pdf:

        for p in output_mc:
            if p in x_label:
                Plots.PlotTagged(output_mc[p]["value"], output_mc[p]["tags"], data2 = output_data[p]["value"] if output_data else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.5, color = "C6")
                Plots.plt.ylim(bottom = 0)
                pdf.Save()
                if f"{p}_completeness" in output_mc:
                    Plots.PlotHist2DImshowMarginal(ak.ravel(output_mc[p]["value"]), ak.ravel(output_mc[f"{p}_completeness"]["value"]), ylabel = "Completeness", xlabel = x_label[p], x_range = x_range[p], bins = nbins[p], norm = "column", c_scale = "linear")
                    Plots.DrawMultiCutPosition(output_mc[p]["cuts"], face = output_mc[p]["op"], arrow_length = CalculateArrowLength(output_mc[p]["value"], x_range[p]), arrow_loc = 0.1, color = "C6")
                    pdf.Save()
        Plots.PlotTags(output_mc["final_tags"]["tags"], xlabel = "True particle ID")
        pdf.Save()
    Plots.plt.close("all")
    return


def MakePFOSelectionPlotsConsdensed(output_mc : dict, output_mc_loose : dict, output_data : dict, output_data_loose : dict, outDir : str, norm : float, book_name : str):
    norm = False if output_data is None else norm

    with Plots.PlotBook(outDir + book_name) as pdf:

        for p in output_mc_loose:
            if p in x_label:
                Plots.PlotTagged(output_mc_loose[p]["value"], output_mc_loose[p]["tags"], data2 = output_data_loose[p]["value"] if output_data_loose else None, norm = norm, y_scale = y_scale[p], x_label = x_label[p], bins = nbins[p], ncols = ncols[p], x_range = x_range[p], truncate = truncate[p])
                
                for c, mc in zip(["C6", "magenta"], [output_mc, output_mc_loose]):
                    if p == "PiPlusSelection":
                        sf = 0.5
                    else:
                        sf = 1
                    Plots.DrawMultiCutPosition(mc[p]["cuts"], face = mc[p]["op"], arrow_length = sf * CalculateArrowLength(mc[p]["value"], x_range[p]), arrow_loc = 0.5, color = c)
                Plots.plt.ylim(bottom = 0)                
                pdf.Save()
                if f"{p}_completeness" in output_mc:
                    Plots.PlotHist2DImshowMarginal(ak.ravel(mc[p]["value"]), ak.ravel(mc[f"{p}_completeness"]["value"]), ylabel = "Completeness", xlabel = x_label[p], x_range = x_range[p], bins = nbins[p], norm = "column", c_scale = "linear")

                    for c, mc in zip(["C6", "magenta"], [output_mc, output_mc_loose]):
                        Plots.DrawMultiCutPosition(mc[p]["cuts"], face = mc[p]["op"], arrow_length = CalculateArrowLength(mc[p]["value"], x_range[p]), arrow_loc = 0.1, color = c)

                    pdf.Save()

        for mc in [output_mc, output_mc_loose]:
            Plots.PlotTags(mc["final_tags"]["tags"], xlabel = "True particle ID")
            pdf.Save()
    Plots.plt.close("all")
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

    with Plots.PlotBook(outDir + "pi0.pdf") as pdf:
        if "NPhotonCandidateSelection" in output_mc:
            if output_data is not None:
                scale = ak.count(output_data["NPhotonCandidateSelection"]["value"]) / ak.count(output_mc["NPhotonCandidateSelection"]["value"])

                n_photons_scaled = []
                u, c = np.unique(output_mc["NPhotonCandidateSelection"]["value"], return_counts = True)
                for i, j in zip(u, c):
                    n_photons_scaled.extend([i]* int(scale * j))

                Plots.PlotBarComparision(n_photons_scaled, output_data["NPhotonCandidateSelection"]["value"], xlabel = "Number of $\\pi^{0}$ photon candidates", label_1 = "MC", label_2 = "Data", fraction = True, barlabel = False)
            else:
                Plots.PlotBar(output_mc["NPhotonCandidateSelection"]["value"], xlabel = "Number of $\\pi^{0}$ photon candidates")
            pdf.Save()

        if "Pi0MassSelection" in output_mc:
            Plots.PlotTagged(output_mc["Pi0MassSelection"]["value"], output_mc["Pi0MassSelection"]["tags"], data2 = output_data["Pi0MassSelection"]["value"] if output_data else None, bins = nbins, x_label = "$m_{\\gamma\\gamma}$ (MeV)", x_range = [0, 500], norm = norm, ncols = 1)
            Plots.DrawMultiCutPosition(output_mc["Pi0MassSelection"]["cuts"], face = output_mc["Pi0MassSelection"]["op"], arrow_length = 50, color = "C6")
            pdf.Save()

            Plots.PlotTagged(output_mc["Pi0MassSelection"]["value"], output_mc["mass_event_tag"]["tags"], data2 = output_data["Pi0MassSelection"]["value"] if output_data else None, bins = nbins, x_label = "$m_{\\gamma\\gamma}$ (MeV)", x_range = [0, 500], norm = norm, ncols = 1)
            Plots.DrawMultiCutPosition(output_mc["Pi0MassSelection"]["cuts"], face = output_mc["Pi0MassSelection"]["op"], arrow_length = 50, color = "C6")
            pdf.Save()

        if "Pi0OpeningAngleSelection" in output_mc:
            Plots.PlotTagged(output_mc["Pi0OpeningAngleSelection"]["value"], output_mc["Pi0OpeningAngleSelection"]["tags"], data2 = output_data["Pi0OpeningAngleSelection"]["value"] if output_data else None, bins = nbins, x_label = "$\\phi$ (rad)", norm = norm, ncols = 1)

            Plots.DrawMultiCutPosition((np.array(output_mc["Pi0OpeningAngleSelection"]["cuts"]) * np.pi / 180).tolist(), face = output_mc["Pi0OpeningAngleSelection"]["op"], arrow_length = 0.25, color = "C6")
            Plots.plt.ylim(bottom = 0)
            pdf.Save()
    Plots.plt.close("all")
    return


def MakeRegionPlots(outputs_mc_masks : dict, outputs_data_masks : dict, outDir : str):
    """ Correlation matrices for truth and reco region selection.

    Args:
        outputs_mc_masks (dict): mc masks for each region
        outputs_data_masks (dict): data masks for each region
        outDir (str): output directory
    """
    with Plots.PlotBook(outDir + "regions.pdf") as pdf:
        # Visualise the regions
        Plots.plot_region_data(outputs_mc_masks["truth_regions"], compare_max=0, title="truth regions")
        pdf.Save()
        Plots.plot_region_data(outputs_mc_masks["reco_regions"], compare_max=0, title="reco regions")
        pdf.Save()
        # Compare the regions
        Plots.compare_truth_reco_regions(outputs_mc_masks["reco_regions"], outputs_mc_masks["truth_regions"], title="")
        pdf.Save()

        if outputs_data_masks is not None:
            Plots.plot_region_data(outputs_data_masks["reco_regions"], compare_max=0, title="reco regions")
            pdf.Save()
    Plots.plt.close("all")
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
