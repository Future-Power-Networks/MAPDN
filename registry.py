from collections import namedtuple
from models.maddpg import MADDPG
from models.sqddpg import SQDDPG
from models.iac import IAC
from models.iddpg import IDDPG
from models.coma import COMA
from models.maac import MAAC
from models.matd3 import MATD3
from models.ippo import IPPO
from models.mappo import MAPPO


Model = dict(maddpg=MADDPG,
             sqddpg=SQDDPG,
             iac=IAC,
             iddpg=IDDPG,
             coma=COMA,
             maac=MAAC,
             matd3=MATD3,
             ippo=IPPO,
             mappo=MAPPO
            )

Strategy=dict(maddpg='pg',
              sqddpg='pg',
              iac='pg',
              iddpg='pg',
              coma='pg',
              maac='pg',
              matd3='pg',
              ippo='pg',
              mappo='pg'
             )
