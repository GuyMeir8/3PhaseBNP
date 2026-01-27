from abc import ABC
from typing import List, Dict, Callable, Optional
import numpy as np

class InteractionPhaseData:
    """
    Class specifically for interaction phase data (Li parameters).
    """
    def __init__(self, Li: List[Callable[[float], float]]):
        self.Li = Li

    def get_Li_per_T(self, T: float) -> np.ndarray:
        return np.array([f(T) for f in self.Li])

class BaseInteraction(ABC):
    """
    Abstract Base Class for Material Interactions.
    """
    def __init__(self):
        self.phase_names: List[str] = []
        self.phases: Dict[str, InteractionPhaseData] = {}

class AgCuData(BaseInteraction):
    def __init__(self):
        super().__init__()
        self.name = "AgCu"
        self.phase_names = ['FCC', 'Liquid']
        self.phases = {
            'FCC' : InteractionPhaseData(
                Li= [
                    lambda T: 36772.58 - 11.02847*T,
                    lambda T: -4612.43 + 0.28869*T,
                    lambda T: 0,
                    lambda T: 0
                ]
            ),
            'Liquid' : InteractionPhaseData(
                Li= [
                    lambda T: 17384.37 - 4.46438 * T,
                    lambda T: 1660.74 - 2.31516 * T,
                    lambda T: 0,
                    lambda T: 0
                ]
            )
        }