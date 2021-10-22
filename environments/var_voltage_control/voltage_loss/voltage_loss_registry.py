from .bowl import bowl_loss
from .bump import bump_loss
from .courant_beltrami import courant_beltrami_loss
from .l1 import l1_loss
from .l2 import l2_loss



Voltage_loss = dict(
    l1=l1_loss,
    l2=l2_loss,
    bowl=bowl_loss,
    bump=bump_loss,
    courant_beltrami=courant_beltrami_loss
)