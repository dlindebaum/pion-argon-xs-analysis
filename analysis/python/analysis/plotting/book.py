import warnings

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

class PlotBook:
    """Manage a multipage PDF destination for saving Matplotlib figures.

        Args:
            name (str): Output PDF path; ``.pdf`` is appended when absent.
            open (bool): Open the PDF destination during initialization.
            watermark (str): Text to place over each saved figure, if provided.
    """
    def __init__(self, name : str, open : bool = True, watermark : str = None) -> None:
        """Set the output path and watermark, and optionally open the PDF.

            Args:
                name (str): Output PDF path; ``.pdf`` is appended when absent.
                open (bool): Open the PDF destination during initialization.
                watermark (str): Text to place over each saved figure, if provided.
        """
        self.name = name
        self.watermark = watermark
        if ".pdf" not in self.name: self.name += ".pdf" 
        if open: self.open()
        self.is_open = True

    def __enter__(self):
        """Return the plot book for use as a context manager.

            Returns:
                Any: This plot book.
        """
        return self

    def __exit__(self, type, value, traceback):
        """Close the PDF when leaving a context manager block.

            Args:
                type (Any): Exception type, if the block raised an exception.
                value (Any): Exception instance, if one was raised.
                traceback (Any): Traceback for the raised exception, if any.
        """
        self.close()
        self.is_open = False

    def watermark(self):
        """Draw the configured watermark at the center of the current Matplotlib axes.
        """
        plt.text(0.5, 0.5, self.watermark, transform=plt.gca().transAxes, fontsize=38, color='gray', alpha=0.5, ha='center', va='center', rotation=30, zorder = np.inf)
        return

    def save(self):
        """Append the current figure to the open PDF, attempting to call the watermark attribute when configured.

            Do nothing when no PDF writer is open; ignore ``AttributeError`` raised during saving.
        """
        global preliminary
        if hasattr(self, "pdf"):
            try:
                if self.watermark is not None:
                    self.watermark()
                self.pdf.savefig(bbox_inches='tight')
            except AttributeError:
                pass

    def open(self):
        """Open the configured output path as a multipage PDF, unless already open.
        """
        if not hasattr(self, "pdf"):
            self.pdf = PdfPages(self.name)
            print(f"pdf {self.name} has been opened")
        else:
            warnings.warn("pdf has already been opened")
        return

    def close(self):
        """Close the multipage PDF and remove its writer, warning if it was not open.
        """
        if hasattr(self, "pdf"):
            self.pdf.close()
            delattr(self, "pdf")
            print(f"pdf {self.name} has been closed")
        else:
            warnings.warn("pdf has not been opened.")
        return

    @classmethod
    def null(cls):
        """Create a plot book with no output file opened.

            Returns:
                Any: A ``PlotBook`` instance initialized with an empty name and ``open=False``.
        """
        return cls(name = "", open = False)
