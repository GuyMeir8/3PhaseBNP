import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, List, Optional

from system_data import SystemData
from BNP_Gibbs_en_calc_3_phase import GibbsEnergyCalculator3Phase

@dataclass
class OptimizationResult3Phase:
    """Stores the result of a 3-phase energy minimization calculation."""
    G_min: float
    A_ratio_alpha: float
    B_ratio_alpha: float
    geometry_type: str
    primary_phases: Tuple[str, str]
    has_skin: bool
    xB_skin: float
    n_alpha: float = 0.0
    n_beta: float = 0.0
    xB_alpha: float = 0.0
    xB_beta: float = 0.0
    r_vals: List[float] = None

class BNPOptimizer3Phase:
    """
    Finds the minimum Gibbs free energy for a given system configuration
    by optimizing the distribution of materials between phases.
    """
    def __init__(self, system_data: SystemData):
        """
        Initializes the optimizer with system data and the energy calculator.
        """
        self.system_data = system_data
        self.calculator = GibbsEnergyCalculator3Phase(system_data)

    def find_minimum_energy(self,
                            T: float,
                            n_total: float,
                            xB_total: float,
                            primary_phases: Tuple[str, str],
                            geometry_type: str,
                            has_skin: bool,
                            xB_skin_guess: float = 0.5,
                            initial_guess: List[float] = [0.5, 0.5]
                            ) -> OptimizationResult3Phase:
        """
        Optimizes the material distribution (A_ratio_alpha, B_ratio_alpha)
        to find the minimum Gibbs free energy for a specific configuration,
        as well as optionally optimizing the skin composition if has_skin is True.

        Returns an OptimizationResult3Phase object with the results.
        """

        def objective(ratios: List[float]) -> float:
            """
            The objective function to be minimized. It calculates the total
            Gibbs energy per mole.
            """
            A_ratio_alpha = ratios[0]
            B_ratio_alpha = ratios[1]
            
            # If optimizing for skin, the 3rd parameter is xB_skin
            # Otherwise, we use the provided guess (though it's unused in calc if has_skin=False)
            current_xB_skin = ratios[2] if has_skin else xB_skin_guess
            
            # The calculator returns the total Gibbs energy. We normalize by n_total
            # for the optimizer to make the function landscape more stable across
            # different particle sizes (n_total).
            G_total = self.calculator.calculate_total_energy(
                A_ratio_alpha=A_ratio_alpha,
                B_ratio_alpha=B_ratio_alpha,
                T=T,
                n_total=n_total,
                xB_total=xB_total,
                primary_phases=primary_phases,
                geometry_type=geometry_type,
                has_skin=has_skin,
                xB_skin=current_xB_skin
            )
            
            # The calculator returns 1.0 on failure, which is a high value for g_per_mole.
            if G_total == 1.0:
                return 1.0
            
            return G_total / n_total

        # Define constraints
        cons = []
        if geometry_type == "Core_Shell":
            def shell_thickness_constraint(ratios: List[float]) -> float:
                """
                Ensures the shell (Beta phase) has enough moles to form at least
                a monoatomic layer around the core (Alpha phase).
                """
                A_ratio_alpha = ratios[0]
                B_ratio_alpha = ratios[1]
                current_xB_skin = ratios[2] if has_skin else xB_skin_guess
                
                if has_skin:
                    phases = primary_phases + ("Liquid",)
                else:
                    phases = primary_phases

                try:
                    n_mp, x_mp, _ = self.calculator._calc_mole_splits_and_geo(
                        A_ratio_alpha, 
                        B_ratio_alpha,
                        T, 
                        n_total, 
                        xB_total, 
                        phases,
                        geometry_type,
                        has_skin,
                        current_xB_skin
                    )
                except ValueError:
                    # If calculation fails, treat as constraint violation
                    return -1.0

                v_mp = self.calculator._get_v_mp(T, phases)
                
                # Core (Alpha)
                V_core = np.sum(n_mp[:, 0] * v_mp[:, 0])
                r_core = self.calculator.calc_r_from_V(V_core)
                
                # Shell (Beta)
                V_shell = np.sum(n_mp[:, 1] * v_mp[:, 1])
                xB_shell = x_mp[1, 1]
                
                mats = self.system_data.config.materials
                r_A = self.system_data.material_data[mats[0]].atomic_radius
                r_B = self.system_data.material_data[mats[1]].atomic_radius
                
                # Weighted thickness (Diameter)
                t_min = 2 * ((1 - xB_shell) * r_A + xB_shell * r_B)
                
                V_shell_min = (4/3) * np.pi * ((r_core + t_min)**3 - r_core**3)
                
                return V_shell - V_shell_min
            
            cons.append({'type': 'ineq', 'fun': shell_thickness_constraint})

        # Ratios must be between 0 and 1, but not exactly 0 or 1 to avoid division by zero.
        eps = 1e-4
        
        if has_skin:
            # Optimize: A_ratio, B_ratio, xB_skin
            bounds = [(eps, 1.0 - eps), (eps, 1.0 - eps), (eps, 1.0 - eps)]
        else:
            # Optimize: A_ratio, B_ratio
            bounds = [(eps, 1.0 - eps), (eps, 1.0 - eps)]

        # Use multiple initial guesses to increase the chance of finding the global minimum.
        base_guesses = [
            initial_guess,
            [0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1], [0.5, 0.5]
        ]
        
        initial_guesses = []
        if has_skin:
            # Use xB_total as the guess for xB_skin
            for bg in base_guesses:
                initial_guesses.append(list(bg) + [xB_total])
        else:
            initial_guesses = base_guesses

        # Remove duplicates by converting to a set of tuples and back.
        initial_guesses = [list(x) for x in set(tuple(x) for x in initial_guesses)]

        best_g_per_mole = float('inf')
        best_ratios = None

        for guess in initial_guesses:
            sol = minimize(fun=objective, x0=guess, method='SLSQP', bounds=bounds, constraints=cons)

            if sol.success and sol.fun < best_g_per_mole:
                best_g_per_mole = sol.fun
                best_ratios = sol.x

        if best_ratios is None:
            # This indicates that the optimization failed for all initial guesses.
            return OptimizationResult3Phase(
                G_min=float('inf'), A_ratio_alpha=float('nan'), B_ratio_alpha=float('nan'),
                geometry_type=geometry_type, primary_phases=primary_phases,
                has_skin=has_skin, xB_skin=xB_skin_guess,
                n_alpha=float('nan'), n_beta=float('nan'), xB_alpha=float('nan'), xB_beta=float('nan'), r_vals=[]
            )

        if has_skin:
            A_ratio_at_min, B_ratio_at_min, xB_skin_at_min = best_ratios
            phases = primary_phases + ("Liquid",)
        else:
            A_ratio_at_min, B_ratio_at_min = best_ratios
            xB_skin_at_min = xB_skin_guess
            phases = primary_phases
        
        # Recalculate geometry and mole splits at the minimum to get detailed results
        try:
            n_mp, x_mp, r_vals = self.calculator._calc_mole_splits_and_geo(
                A_ratio_at_min, 
                B_ratio_at_min,
                T, 
                n_total, 
                xB_total, 
                phases,
                geometry_type,
                has_skin,
                xB_skin_at_min
            )
            n_alpha = np.sum(n_mp[:, 0])
            n_beta = np.sum(n_mp[:, 1])
            xB_alpha = x_mp[1, 0]
            xB_beta = x_mp[1, 1]
            r_vals_list = r_vals.tolist()
        except ValueError:
            n_alpha, n_beta, xB_alpha, xB_beta = float('nan'), float('nan'), float('nan'), float('nan')
            r_vals_list = []

        # The optimizer worked with G/n_total, so we scale it back up for the final result.
        G_min = best_g_per_mole * n_total

        return OptimizationResult3Phase(
            G_min=G_min, A_ratio_alpha=A_ratio_at_min, B_ratio_alpha=B_ratio_at_min,
            geometry_type=geometry_type, primary_phases=primary_phases,
            has_skin=has_skin, xB_skin=xB_skin_at_min,
            n_alpha=n_alpha, n_beta=n_beta, xB_alpha=xB_alpha, xB_beta=xB_beta, r_vals=r_vals_list
        )
