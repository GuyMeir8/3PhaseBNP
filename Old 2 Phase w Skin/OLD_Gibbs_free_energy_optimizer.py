import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, List, Optional
from generic_configuration_gibbs_free_energy import GenericConfiguration, GeometryType, GeometrySettings
from Gibbs_energy_calculator import GibbsEnergyCalculator


@dataclass
class OptimizationResult:
    """Stores the result of a calculation for a specific geometry/phase combo."""
    G_min: float
    geometry: Optional[GeometrySettings]
    phase_pair: Optional[Tuple[str, str]]

    # Defaults
    ratio_A_alpha: float = 1.0
    ratio_B_alpha: float = 1.0
    is_single_phase: bool = False

    # Physical Properties
    n_alpha: float = 0.0
    x_B_alpha: float = 0.0
    n_beta: float = 0.0
    x_B_beta: float = 0.0


class GibbsOptimizer:
    def __init__(self, config: GenericConfiguration):
        self.config = config
        self.calculator = GibbsEnergyCalculator(config)

    def solve_specific_configuration(self,
                                     T: float,
                                     xB_total: float,
                                     n_total: float,
                                     geometry: GeometrySettings,
                                     phase_pair: Tuple[str, str],
                                     initial_guess: List[float] = [0.5, 0.5]) -> OptimizationResult:
        """
        Calculates Gibbs energy for ONE specific geometry and phase pair combination.
        """

        # --- CASE A: SINGLE PHASE ---
        if geometry.geometry_type == GeometryType.SINGLE_PHASE:
            phase = phase_pair[0]
            try:
                # Unpack new return values
                G_val, n_core, n_shell, x_core = self.calculator.calculate_single_phase_gibbs_free_energy(
                    n_total=n_total, xB_total=xB_total, T=T,
                    phase=phase, geometry=geometry
                )

                # Map Core -> Alpha, Shell -> Beta (if exists)
                # If no shell, n_beta will be 0
                return OptimizationResult(
                    G_min=G_val,
                    geometry=geometry,
                    phase_pair=phase_pair,
                    is_single_phase=True,
                    n_alpha=n_core,
                    x_B_alpha=x_core,
                    n_beta=n_shell,
                    # If shell exists, x_B is either 0 (Mat A) or 1 (Mat B)
                    x_B_beta=1.0 if (geometry.has_outer_shell and geometry.shell.material == 1) else 0.0
                )
            except Exception:
                return OptimizationResult(G_min=float('inf'), geometry=geometry, phase_pair=phase_pair)

        # --- CASE B: TWO PHASE ---
        else:
            return self._optimize_two_phase(T, xB_total, n_total, phase_pair, geometry, initial_guess)

    def _optimize_two_phase(self, T, xB_total, n_total, phase_pair, geometry, initial_guess) -> OptimizationResult:

        initial_guesses = [
            initial_guess,
            [0.1, 0.1],
            [0.9, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.5, 0.5]
        ]
        initial_guesses = [list(x) for x in set(tuple(x) for x in initial_guesses)]

        if geometry.geometry_type == GeometryType.CORE_SHELL:

            # 1. Calculate Physical Minimum Constraint
            # We need the minimum thickness to be the atomic radius
            r_A = self.calculator.pure_mat_data[0].atomic_radius
            r_B = self.calculator.pure_mat_data[1].atomic_radius
            min_thickness = max(r_A, r_B)

            # Estimate Volume properties
            v_avg = np.average(self.calculator.calc_v_mp(T, phase_pair))

            V_total = n_total * v_avg
            R_total = ((3 * V_total) / (4 * np.pi)) ** (1 / 3)

            # Minimum physical moles = Volume of monolayer / Molar Volume
            V_min_layer = (4 * np.pi * (R_total ** 2)) * min_thickness
            N_min_moles = V_min_layer / v_avg

            # 2. Define Constraint Function
            # Constraint: n_alpha >= N_min  AND  n_beta >= N_min
            def constraint_min_moles(ratios):
                r_A, r_B = ratios

                # Use calculator to Determine n_alpha and n_beta.
                # This function AUTOMATICALLY subtracts the shell moles if geometry.has_outer_shell is True.
                n_mp, _ = self.calculator.calc_n_x(
                    ratio_A_alpha=r_A,
                    ratio_B_alpha=r_B,
                    n_total=n_total,
                    xB_total=xB_total,
                    T=T,
                    phase_pair=phase_pair,
                    geometry=geometry
                )

                # Sum moles for each phase (column 0 = Alpha, column 1 = Beta)
                n_alpha = np.sum(n_mp[:, 0])
                n_beta = np.sum(n_mp[:, 1])

                # Return minimum excess moles (must be positive to satisfy constraint)
                return min(n_alpha - N_min_moles, n_beta - N_min_moles)

            cons = {'type': 'ineq', 'fun': constraint_min_moles}
        else:
            cons = False


        # 3. Optimization
        def objective(ratios):
            r_A, r_B = ratios
            G_pre_normalize = self.calculator.calculate_total_Gibbs_free_energy(
                ratio_A_alpha=r_A, ratio_B_alpha=r_B,
                n_total=n_total, xB_total=xB_total, T=T,
                phase_pair=phase_pair, geometry=geometry
            )
            return G_pre_normalize / n_total

        eps = 1e-4
        bounds = [(eps, 1.0 - eps), (eps, 1.0 - eps)]





        G0_min = 1.0
        r_A_at_min = 0.5
        r_B_at_min = 0.5

        try:
            for initial_guess_curr in initial_guesses:

                if cons == False:
                    optimizer_args = {
                        'fun': objective,
                        'x0': initial_guess_curr,
                        'method': 'SLSQP',
                        'bounds': bounds
                    }
                else:
                    optimizer_args = {
                        'fun': objective,
                        'x0': initial_guess_curr,
                        'method': 'SLSQP',
                        'bounds': bounds,
                        'constraints': cons
                    }
                sol = minimize(**optimizer_args)

                if G0_min > sol.fun:
                    G0_min = sol.fun
                    r_A_at_min, r_B_at_min = sol.x[0], sol.x[1]


            n_mp, x_mp = self.calculator.calc_n_x(
                ratio_A_alpha=r_A_at_min,
                ratio_B_alpha=r_B_at_min,
                n_total=n_total,
                xB_total=xB_total,
                T=T,
                phase_pair=phase_pair,
                geometry=geometry
            )

            n_alpha_total = np.sum(n_mp[:, 0])
            n_beta_total = np.sum(n_mp[:, 1])

            x_B_alpha = x_mp[1, 0]
            x_B_beta = x_mp[1, 1]

            # if geometry.geometry_type == GeometryType.CORE_SHELL and not geometry.has_outer_shell:
            #     core_shell_threshold = 1e-3
            #     if x_B_beta > 1 - core_shell_threshold or x_B_beta < core_shell_threshold:
            #         return OptimizationResult(G_min=1.0, geometry=geometry, phase_pair=phase_pair)

            return OptimizationResult(
                G_min= n_total * G0_min,
                geometry=geometry,
                phase_pair=phase_pair,
                ratio_A_alpha=r_A_at_min,
                ratio_B_alpha=r_B_at_min,
                is_single_phase=False,
                n_alpha=n_alpha_total,
                x_B_alpha=x_B_alpha,
                n_beta=n_beta_total,
                x_B_beta=x_B_beta
            )
        except Exception as e:
            print(f"Optimization failed for {geometry.geometry_type} {phase_pair} T={T} xB_total={xB_total} n_total={n_total}: {e}")
            return OptimizationResult(G_min=float('inf'), geometry=geometry, phase_pair=phase_pair)