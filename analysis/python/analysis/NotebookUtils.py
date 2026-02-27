"""
Created on: 21/01/2024 17:14

Author: Shyam Bhuller

Description: random functions which are sometimes useful in notebooks
"""

import sys

from IPython.core.magic import register_line_magic
from IPython import get_ipython

@register_line_magic
def init_notebook(line):
    ipython = get_ipython()
    ipython.run_line_magic("matplotlib", "inline") # %matplotlib inline
    ipython.run_line_magic("load_ext", "autoreload") # %load_ext autoreload
    ipython.run_line_magic("autoreload", "2") # %autoreload 2

    #* magic to add python path to notebook environment
    for pypath in sys.path:
        if pypath.split("/")[-1] == "analysis": break
    print(pypath)
    ipython.run_line_magic("env", "PYTHONPATH $pypath") # %env PYTHONPATH $pypath
    return
