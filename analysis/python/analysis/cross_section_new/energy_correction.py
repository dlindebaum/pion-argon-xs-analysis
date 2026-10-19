import numpy as np
import Fitting

def LinearCorrection(x, p0):
    return x / p0

class ResponseFit(Fitting.FitFunction):
    n_params = 3

    @staticmethod
    def func(x, p0, p1, p2):
        return p0 * np.log(x - p1) + p2

    @staticmethod
    def p0(x, y):
        return None

def ResponseCorrection(x, p0, p1, p2):
    return x / (ResponseFit.func(x, p0, p1, p2) + 1)


shower_energy_correction = {
    "linear" : LinearCorrection,
    "response": ResponseCorrection,
    None : None
}

