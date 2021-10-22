import numpy as np



def l2_loss(vs, v_ref=1.0):
    def _l2_loss(v):
        return 2 * np.square(v - v_ref)
    return np.array([_l2_loss(v) for v in vs])