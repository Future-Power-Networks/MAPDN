import numpy as np



def courant_beltrami_loss(vs, v_lower, v_upper):
    def _courant_beltrami_loss(v):
        return np.square(max(0, v - v_upper)) + np.square(max(0, v_lower - v))
    return np.array([_courant_beltrami_loss(v) for v in vs])