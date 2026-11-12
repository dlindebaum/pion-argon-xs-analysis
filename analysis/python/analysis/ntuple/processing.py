from argparse import Namespace

from python.analysis import processing
from . import ntuple

def calculate_batches(args : Namespace) -> dict[str, any]:
    """Build the multiprocessing options from the application arguments.

        Args:
            args (Namespace): Parsed application arguments containing the input.
                Ntuple files and CPU count. The current implementation leaves
                event and batch limits unset and uses ``args.cpus`` as threads.

        Returns:
            dict[str, any]: Processing options with ``events`` and ``batches``
                set to ``None`` and ``threads`` set to the requested CPU count.
    """
    if "data" in args.ntuple_files:
        n_data = [ntuple.file_len(file_desc.file) for file_desc in args.ntuple_files["data"]]
    else:
        n_data = []

    if len(n_data) == 0:
        print("no data file was specified, 'normalisation', 'beam_reweight', 'toy_parameters' and 'analyse' will not run")

    n_mc = [ntuple.file_len(file_desc.file) for file_desc in args.ntuple_files["mc"]] # must have MC

    processing_args = {"events" : None, "batches" : None, "threads" : args.cpus}

    # pass multiprocessing args
    # if max([*n_data, *n_mc]) >= 7E5:
    #     processing_args["events"] = None
    #     processing_args["batches"] = int(2 * max([*n_data, *n_mc]) // 7E5)
    #     processing_args["threads"] = args.cpus

    return processing_args


def run_process(ntuple_files : list[ntuple.FileDescriptor], is_data : bool, args : Namespace, func : callable, merge : bool = True) -> list[dict] | dict:
    """Run a processing function over ntuple files and optionally merge its outputs.

        Args:
            ntuple_files (list[ntuple.FileDescriptor]): Ntuple files.
            is_data (bool): Whether the Ntuple files are Data or MC.
            args (Namespace): Application arguments.
            func (callable): Process to run.
            merge (bool): Whether to merge the outputs of the processing. Defaults to True.

        Returns:
            list[dict] | dict: Per-file processing outputs when ``merge`` is.
                False, or the dictionary returned by
                ``processing.merge_outputs`` when ``merge`` is true.
    """
    func_args = vars(args)
    func_args["data"] = is_data
    output = processing.mutliprocess(func, ntuple_files, args.batches, args.events, func_args, args.threads)
    if merge:
        output = processing.merge_outputs(output)
    return output
