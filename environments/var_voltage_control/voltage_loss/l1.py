import numpy as np



def l1_loss(vs, v_ref=1.0):
    def _l1_loss(v):
        return np.abs( v - v_ref )
    return np.array([_l1_loss(v) for v in vs])