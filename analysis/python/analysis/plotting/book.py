import warnings

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

class PlotBook:
    def __init__(self, name : str, open : bool = True, watermark : str = None) -> None:
        self.name = name
        self.watermark = watermark
        if ".pdf" not in self.name: self.name += ".pdf" 
        if open: self.open()
        self.is_open = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        self.is_open = False

    def PlotWatermark(self):
        plt.text(0.5, 0.5, self.watermark, transform=plt.gca().transAxes, fontsize=38, color='gray', alpha=0.5, ha='center', va='center', rotation=30, zorder = np.inf)
        return

    def Save(self):
        global preliminary
        if hasattr(self, "pdf"):
            try:
                if self.watermark is not None:
                    self.PlotWatermark()
                self.pdf.savefig(bbox_inches='tight')
            except AttributeError:
                pass

    def open(self):
        if not hasattr(self, "pdf"):
            self.pdf = PdfPages(self.name)
            print(f"pdf {self.name} has been opened")
        else:
            warnings.warn("pdf has already been opened")
        return

    def close(self):
        if hasattr(self, "pdf"):
            self.pdf.close()
            delattr(self, "pdf")
            print(f"pdf {self.name} has been closed")
        else:
            warnings.warn("pdf has not been opened.")
        return

    @classmethod
    def null(cls):
        return cls(name = "", open = False)
