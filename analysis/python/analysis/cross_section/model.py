"""Cross-section model-building logic."""

from python.analysis import slices, samples

def make_systematic(proc_info : tuple, init_bin : tuple, end_bin : tuple):
    init_low, init_high = init_bin[1]
    end_low, end_high = end_bin[1]
    z_low, z_high = proc_info[1]["z_region"]
    mode = proc_info[1]["mode"]

    name = f"True{proc_info[0]}-Init_{init_bin[0]}-End_{end_bin[0]}"

    return  f"""
- Systematic:
    Names:
      FancyName: {name}
      ParameterName: {name}

    SampleNames: ["PDSP"]
    Mode: [{mode}]
    Error: 0.1
    FlatPrior: true
    ParameterBounds: [0, 4]
    ParameterGroup: Fit
    KinematicCuts:
      - TrueKEInt: [{end_low}, {end_high}]
      - TrueKEIni: [{init_low}, {init_high}]
      - TrueEndZ: [{z_low}, {z_high}]
    ParameterValues:
      Generator: 1.
      PreFitValue: 1.
    Type: Norm
    StepScale:
      MCMC: 0.01
"""

def slices_to_model_bins(energy_slices : slices.Slices) -> dict[tuple]:
    model_bins = [0] + list(energy_slices.edges[::-1]) + [999999]
    # model_bins = np.where(model_bins == energy_slices.underflow_pos, 0, model_bins)
    # model_bins = np.where(model_bins == energy_slices.overflow_pos, 999999, model_bins)

    bins = {}
    for i in range(len(model_bins)-1):
        if i == 0:
            k = "Under"
        elif i == len(model_bins)-2:
            k = "Over"
        else:
            k = str(i-1)
        bins[k] = (model_bins[i], model_bins[i+1])
    return bins


def get_process_info(process : samples.SampleDefinition, fiducial_volume : list[float]) -> dict:
    model_process_info = {}

    for k, v in process.definitions.items():
        definitions =  process.convert_process_definitions_str(str(v[0]))
        if "beam_escapes" in definitions:
            if definitions["beam_escapes"]["op"] == "==" and definitions["beam_escapes"]["value"] == 0: # beam particle does not escape, so events for this process must be within the fv.
                fv = fiducial_volume
            elif definitions["beam_escapes"]["op"] == "==" and definitions["beam_escapes"]["value"] == 1: # beam particle escapes fv for this sample.
                fv = [max(fiducial_volume), 999]
            else:
                fv = [min(fiducial_volume), 999] # ends before fv.
        else:
            fv = (0, 999) # if beam_escapes is not included, then disregard the fv.
        model_process_info[process.short_name[k]] = {"mode" : process.mode[k], "z_region" : fv}
    return model_process_info

