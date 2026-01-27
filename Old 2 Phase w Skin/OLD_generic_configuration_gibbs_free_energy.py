import datetime
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Any, Optional
from pure_material_properties import AgData, CuData, BaseMaterial
from surface_energy_calculations import SurfaceEnergyValues
from interaction_properties import AgCuData, BaseInteraction

class GeometryType(Enum):
    """
    Enumeration for different geometric options.
    3.1: GENERAL options like CoreShell and Janus.
    """
    SINGLE_PHASE = "SinglePhase"
    CORE_SHELL = "CoreShell"
    JANUS = "Janus"
    SPHERIC_JANUS = "SphericalJanus"
    # Add other types as needed (e.g. specific experimental imitations)

@dataclass
class ShellSettings:
    """
    Settings for the outer shell material and phase.
    material: 0 for A, 1 for B
    phase: e.g., "FCC" or "Liquid"
    """
    material: int
    phase: str

@dataclass
class GeometrySettings:
    """
    3. Stores geometric configuration.
    Includes the geometry type and the 'with/without outer shell' option (3.2).
    """
    geometry_type: GeometryType
    has_outer_shell: bool = False
    shell: Optional[ShellSettings] = None

@dataclass
class ParticleData:
    """
    4.1 & 4.2: Class to store raw data regarding particles (A or B).
    """
    name: str
    # CHANGE 2: Use specific type hint
    data: Optional[BaseMaterial] = None


@dataclass
class InteractionData:
    """
    4.3: Class to store data regarding interactions between particles (AB).
    """
    name: str
    # CHANGE 3: Use specific type hint
    data: Optional[BaseInteraction] = None


@dataclass
class GenericConfiguration:
    """
    Generic configuration class for Gibbs free energy calculation of a BNP.
    """
    # 1. File name settings
    base_file_name: str

    # 2. Phase options
    phase_options: List[str]

    # 3. Geometric options
    geometry_options: List[GeometrySettings]

    # 4. Data classes
    a_data: ParticleData
    b_data: ParticleData
    ab_data: InteractionData

    # 4.4 xB total values configuration
    xb_step: float

    # Surface energy data (Has default, must come after non-defaults)
    surface_energy_data: SurfaceEnergyValues = field(default_factory=SurfaceEnergyValues)

    xb_edge_cases: List[float] = field(default_factory=list)

    # 5. Temperature values configuration
    t_min: float = 300.0
    t_max: float = 1000.0
    t_step: float = 100.0

    # 6. n_total values (added one by one)
    n_total_values: List[float] = field(default_factory=list)

    def get_output_file_name(self) -> str:
        """
        1. Generates the output csv file name with a timestamp (YearMonthDay_HourMinuteSecond).
        """
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        return f"{self.base_file_name}_{timestamp}.csv"

    @property
    def xb_values(self) -> List[float]:
        """
        4.4 Calculates the list of all x_B_total values.
        Generates values from `step` to `1-step` with `step` jumps.
        Adds edge cases and returns a sorted unique list.
        """
        # We want to range from xb_step to (1 - xb_step)
        start = self.xb_step
        stop = 1.0 - self.xb_step

        # Use numpy for range generation to handle float steps cleanly
        # Adding a small epsilon to stop value to ensure the last point is included if it matches step
        epsilon = self.xb_step / 1000.0
        values = np.arange(start, stop + epsilon, self.xb_step)

        # Round values to avoid floating point artifacts (e.g., 0.30000000004)
        unique_values: Set[float] = {round(x, 6) for x in values}

        # Add edge cases
        for edge in self.xb_edge_cases:
            unique_values.add(edge)

        return sorted(list(unique_values))

    @property
    def temperature_values(self) -> List[float]:
        """
        5. Calculates the list of T values from min to max with defined jumps.
        """
        epsilon = self.t_step / 1000.0
        values = np.arange(self.t_min, self.t_max + epsilon, self.t_step)
        return [round(x, 6) for x in values]


# CHANGE 4: Instantiate classes in the configuration
low_res_configuration = GenericConfiguration(
    base_file_name="low_res_FCC_like_Cu_Liquid_like_Ag",
    phase_options=["FCC", "Liquid"],
    geometry_options=[
        ### GEOMETRIES WITH NO OUTER SHELL ###
        GeometrySettings(geometry_type=GeometryType.CORE_SHELL, has_outer_shell=False),
        GeometrySettings(geometry_type=GeometryType.JANUS, has_outer_shell=False),

        ### CORE SHELL GEOMETRIES WITH OUTER SHELL ###
        # GeometrySettings(
        #     geometry_type=GeometryType.CORE_SHELL,
        #     has_outer_shell=True,
        #     shell=ShellSettings(material=0, phase="FCC")
        # ),
        # GeometrySettings(
        #     geometry_type=GeometryType.CORE_SHELL,
        #     has_outer_shell=True,
        #     shell=ShellSettings(material=1, phase="FCC")
        # ),
        GeometrySettings(
            geometry_type=GeometryType.CORE_SHELL,
            has_outer_shell=True,
            shell=ShellSettings(material=0, phase="Liquid")
        ),
        GeometrySettings(
            geometry_type=GeometryType.CORE_SHELL,
            has_outer_shell=True,
            shell=ShellSettings(material=1, phase="Liquid")
        ),

        ### SPHERIC JANUS GEOMETRIES WITH OUTER SHELL ###
        # GeometrySettings(
        #     geometry_type=GeometryType.SPHERIC_JANUS,
        #     has_outer_shell=True,
        #     shell=ShellSettings(material=0, phase="FCC")
        # ),
        GeometrySettings(
            geometry_type=GeometryType.SPHERIC_JANUS,
            has_outer_shell=True,
            shell=ShellSettings(material=0, phase="Liquid")
        ),
        # GeometrySettings(
        #     geometry_type=GeometryType.SPHERIC_JANUS,
        #     has_outer_shell=True,
        #     shell=ShellSettings(material=1, phase="FCC")
        # ),
        GeometrySettings(
            geometry_type=GeometryType.SPHERIC_JANUS,
            has_outer_shell=True,
            shell=ShellSettings(material=1, phase="Liquid")
        ),

        ### SINGLE PHASE GEOMETRIES WITH/WITHOUT OUTER SHELL ###
        GeometrySettings(geometry_type=GeometryType.SINGLE_PHASE, has_outer_shell=False),
        # GeometrySettings(
        #     geometry_type=GeometryType.SINGLE_PHASE,
        #     has_outer_shell=True,
        #     shell=ShellSettings(material=0, phase="FCC")
        # ),
        # GeometrySettings(
        #     geometry_type=GeometryType.SINGLE_PHASE,
        #     has_outer_shell=True,
        #     shell=ShellSettings(material=1, phase="FCC")
        # ),
        GeometrySettings(
            geometry_type=GeometryType.SINGLE_PHASE,
            has_outer_shell=True,
            shell=ShellSettings(material=0, phase="Liquid")
        ),
        GeometrySettings(
            geometry_type=GeometryType.SINGLE_PHASE,
            has_outer_shell=True,
            shell=ShellSettings(material=1, phase="Liquid")
        ),
    ],
    a_data=ParticleData(name="Ag", data=AgData()),
    b_data=ParticleData(name="Cu", data=CuData()),
    ab_data=InteractionData(name="Ag-Cu", data=AgCuData()),
    surface_energy_data=SurfaceEnergyValues(),
    xb_step=0.01,
    t_min=400.0,
    t_max=1400.0,
    t_step=10.0,
    # n_total_values=[1.0, 5e-17, 5e-19, 5e-21],
    n_total_values=[5e-17, 5e-19, 5e-21],
    # n_total_values=[5e-19],
)