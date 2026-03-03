from math import log
from abc import ABC
from typing import List, Dict, Callable, Optional

class PhaseData:
    """
    Base class for phase data. Can handle either pure material properties (g0, v)
    or interaction parameters (Li).
    """
    def __init__(self,
                 g0: Optional[Callable[[float], float]] = None,
                 v: Optional[Callable[[float], float]] = None
                 ):
        self.g0 = g0
        self.v = v


class BaseMaterial(ABC):
    """
    Abstract Base Class for Pure Materials.
    """
    def __init__(self):
        self.phase_names: List[str] = []
        self.phases: Dict[str, PhaseData] = {}
        self.atomic_radius: float = 0.0
        self.name: str = ""

class AgData(BaseMaterial):
    def __init__(self):
        super().__init__()
        self.phase_names = ['FCC','Liquid']
        self.name = "Ag"
        self.phases = {
            'FCC' : PhaseData(
                g0= lambda T: -7209.512 + 118.202013 * T - 23.8463314 * T * log(T) - 1.790585e-3 * (
                            T ** 2) - 0.398587e-6 * (T ** 3) - 12011 * (T ** (-1))
                if T < 1234.93 else
                -15095.252 + 190.266404*T - 33.472*T*log(T) + 1411.773E26*(T**-9),
                v= lambda T: 9.9361e-6 + T * 1.1368e-9,
            ),
            'Liquid' : PhaseData(
                g0= lambda T : 3815.564+109.310993*T-23.8463314*T*log(T)-1.790585e-3*(T**2)-0.398587e-6*(T**3)-12011*(T**(-1))-1033.905e-23*(T**7)
                if T < 1234.93 else
                -3587.111 + 180.964656*T - 33.472*T*log(T)  ,
                v= lambda T : 1.01961e-5+T*1.1368e-9,
            )
        }
        self.atomic_radius = 160e-12

class CuData(BaseMaterial):
    def __init__(self):
        super().__init__()
        self.name = "Cu"
        self.phase_names = ['FCC','Liquid']
        self.phases = {
            'FCC' : PhaseData(
                g0= lambda T: (
                        -7770.458+130.485235*T-24.112392*T*log(T)-2.65684e-3*T**2+0.129223e-6*T**3+52478*T**(-1)
                if T < 1357.77 else
                -13542.026 + 183.803828*T - 31.38*T*log(T) + 364.167E27*(T**-9)),
                v= lambda T: 7.01e-6+2.92e-10*T+1.02e-13*T**2,
            ),
            'Liquid' : PhaseData(
                g0= lambda T : 5194.277+120.973331*T-24.112392*T*log(T)-2.65684e-3*T**2+0.129223e-6*T**3+52478*T**(-1)-584.89e-23*T**7
                if T < 1357.77 else
                -46.545 + 173.881484*T - 31.38*T*log(T) ,
                v= lambda T : 7.53e-6+2.49e-10*T+1.86e-13*T**2,
            )
        }
        self.atomic_radius = 135e-12


class NiData(BaseMaterial):
    def __init__(self):
        super().__init__()
        self.name = "Ni"
        self.phase_names = ['FCC','Liquid']
        self.phases = {
            'FCC' : PhaseData(
                g0= lambda T: (
                        -5179.159 + 117.854 * T - 22.096 * T * log(T) - 4.8407E-3 * T**2 
                if T <  1728 else
               -27840.620 + 279.134977 * T - 43.1 * T * log(T) + 1127.54E28 * (T**-9)),
                v= lambda T: 6.6e-6,
            ),
            'Liquid' : PhaseData(
                g0= lambda T : 11235.527 + 108.457 * T - 22.096 * T * log(T) - 4.8407E-3 * T**2 - 382.318E-23 * T**7  
                if T < 1728 else
                -9549.817 + 268.597977 * T - 43.1 * T * log(T) ,
                v= lambda T : 5.85e-6 + 9.02e-10*T,
            )
        }
        self.atomic_radius = 124e-12       