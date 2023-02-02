import subprocess

from rich import print

from python.analysis.Processor import Processor

class RunAnalysis(Processor):
    """ Processor class which is setup to run the hit correction analysis.
    """
    def __init__(self):
        super().__init__()

    def Create_CLI(self, description: str = "Process description", eplilog: str = None):
        super().Create_CLI(description, eplilog)
        self.parser.add_argument("-o", "--out-csv", dest = "csv", type = str, default = "out", help = "output csv filename")
        return
        
    def ParseCLI(self):
        super().ParseCLI()
        self.args.csv += ".csv"
        print(vars(self.args))
        return

    def Process(self, file: str, nEvents: int, start: int) -> subprocess.Popen:
        return subprocess.Popen([
            "./true_energy_hit_correction.py",
            file,
            "-o", f"{self.args.csv}", 
            "-e", f"{nEvents}", f"{start}"], 
            stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)

# having main() feels a little extra, but why not?
def main(analysis : Processor):
    analysis.Run()
    return

if __name__ == "__main__":
    analysis = RunAnalysis()
    analysis.Create_CLI("run the hit correction study")
    analysis.ParseCLI()
    main(analysis)
