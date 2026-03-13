import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from dataclasses import dataclass
from typing import Tuple, List, Optional

from configurations_3_phase import ThreePhaseConfiguration
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
    def __init__(self, config: ThreePhaseConfiguration):
        """
        Initializes the optimizer with the configuration and the energy calculator.
        """
        self.config = config
        self.calculator = GibbsEnergyCalculator3Phase(config)

    def find_minimum_energy(self,
                            T: float,
                            n_total: float,
                            xB_total: float,
                            primary_phases: Tuple[str, str],
                            geometry_type: str,
                            has_skin: bool,
                            xB_skin_guess: float = 0.5,
                            initial_guess: Optional[List[float]] = None,
                            exhaustive_search: bool = True
                            ) -> OptimizationResult3Phase:
        """
        Optimizes the material distribution (A_ratio_alpha, B_ratio_alpha)
        to find the minimum Gibbs free energy for a specific configuration,
        as well as optionally optimizing the skin composition if has_skin is True.

        If exhaustive_search is True, initial_guess is ignored (algorithm self-guesses).
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
            skin_val = ratios[2] if has_skin else None
            
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
                skin_val=skin_val # type: ignore
            )
            
            # The calculator returns 1.0 on failure, which is a high value for g_per_mole.
            if G_total == 1.0:
                return 1.0
            
            return G_total / n_total

        # Define constraints
        cons = []
        if geometry_type == "Core Shell":
            def shell_thickness_constraint(ratios: List[float]) -> float:
                """
                Ensures the shell (Beta phase) has enough moles to form at least
                a monoatomic layer around the core (Alpha phase).
                """
                A_ratio_alpha = ratios[0]
                B_ratio_alpha = ratios[1]
                skin_val = ratios[2] if has_skin else None

                try:
                    # Perform necessary setup to call internal calculation methods
                    phases, skin = self.calculator._update_phases_based_on_skin(primary_phases, skin_val)
                    T_dep = self.calculator._get_T_dependent_vars(T, phases)
                    
                    n_mp, x_mp, r_vals = self.calculator._calc_mole_splits_and_geo(
                        A_ratio_alpha, 
                        B_ratio_alpha,
                        n_total, 
                        xB_total, 
                        phases,
                        geometry_type,
                        skin,
                        T,
                        T_dep
                    )
                except ValueError:
                    # If calculation fails, treat as constraint violation
                    return -1.0

                # For Core Shell: r_vals = [r_core, r_outer_shell, (optional r_total_with_skin)]
                r_core = r_vals[0]
                r_shell_outer = r_vals[1]
                
                # Shell (Beta) Composition
                xB_shell = x_mp[1, 1]
                
                mats = self.config.materials
                r_A = self.calculator.material_data[mats[0]].atomic_radius
                r_B = self.calculator.material_data[mats[1]].atomic_radius
                
                # Weighted thickness limit (approx diameter of atom)
                t_min = 2 * ((1 - xB_shell) * r_A + xB_shell * r_B)
                
                current_thickness = r_shell_outer - r_core
                
                return current_thickness - t_min
            
            # For minimize (SLSQP)
            cons.append({'type': 'ineq', 'fun': shell_thickness_constraint})

            # For differential_evolution (Constraint Object)
            # We want shell_thickness_constraint >= 0, so 0 <= val <= inf
            de_constraints = (NonlinearConstraint(shell_thickness_constraint, 0.0, np.inf),)
        else:
            de_constraints = ()

        # --- HELPER: Heuristic Candidates ---
        # These are "smart guesses" to ensure we check specific physical scenarios
        # like complete mixing or phase separation at the solubility limits.
        heuristic_candidates = [
            [0.5, 0.5],          # Mixed
            [0.99, 0.01],        # Phase Separation (A-rich alpha)
            [0.01, 0.99],        # Phase Separation (B-rich alpha)
            [0.01, 0.01],        # Pure Alpha
            [0.99, 0.99],        # Pure Beta
        ]
        if initial_guess is not None:
            heuristic_candidates.insert(0, list(initial_guess))

        def run_local_minimization(start_guess):
            """Helper to run a single local minimization from a guess."""
            guess = list(start_guess)
            if has_skin and len(guess) < 3:
                guess.append(xB_total)
            
            try:
                res = minimize(fun=objective, x0=guess, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-10)
                return res
            except RuntimeError:
                raise # Allow critical errors to propagate
            except:
                return None

        # Ratios must be between 0 and 1, but not exactly 0 or 1 to avoid division by zero.
        eps = 1e-4
        
        if has_skin:
            # Optimize: A_ratio, B_ratio, xB_skin
            bounds = [(eps, 1.0 - eps), (eps, 1.0 - eps), (eps, 1.0 - eps)]
        else:
            # Optimize: A_ratio, B_ratio
            bounds = [(eps, 1.0 - eps), (eps, 1.0 - eps)]

        best_g_per_mole = float('inf')
        best_ratios = None

        # --- STRATEGY 1: GLOBAL OPTIMIZATION (Differential Evolution) ---
        if exhaustive_search:
            # Differential Evolution is robust against non-convex landscapes and step-function failures
            res = differential_evolution(
                func=objective,
                bounds=bounds,
                constraints=de_constraints,
                strategy='best1bin',
                maxiter=50,       # Good balance for this dimensionality
                popsize=15,       # Population per parameter
                tol=0.001,
                polish=False,     # We will polish manually to handle constraints better
                disp=False
            )

            if res.success and res.fun < 1.0: # Check against failure sentinel (1.0)
                best_g_per_mole = res.fun
                best_ratios = res.x
                
                # Manual Polish with Gradient-based solver (SLSQP)
                # This refines the loose solution from DE
                try:
                    res_polish = minimize(
                        fun=objective, 
                        x0=best_ratios, 
                        method='SLSQP', 
                        bounds=bounds, 
                        constraints=cons, 
                        tol=1e-10
                    )
                    if res_polish.success and res_polish.fun < best_g_per_mole:
                        best_g_per_mole = res_polish.fun
                        best_ratios = res_polish.x
                except RuntimeError:
                    raise # Allow critical errors to propagate
                except:
                    pass # If polish fails, keep DE result
            
            # --- AUGMENTED SEARCH: Check Heuristics as well ---
            # Even if DE ran, we double-check the heuristic points to ensure 
            # we didn't miss a narrow basin at the edges (common in phase separation).
            for cand in heuristic_candidates:
                res = run_local_minimization(cand)
                if res and res.success and res.fun < best_g_per_mole:
                     if not (geometry_type == "Core Shell" and shell_thickness_constraint(res.x) < -1e-6):
                        best_g_per_mole = res.fun
                        best_ratios = res.x

        # --- STRATEGY 2: LOCAL OPTIMIZATION (Single Guess) ---
        else:
            # Run through all heuristics instead of just one guess
            for cand in heuristic_candidates:
                res = run_local_minimization(cand)
                if res and res.success and res.fun < best_g_per_mole:
                    if not (geometry_type == "Core Shell" and shell_thickness_constraint(res.x) < -1e-6):
                        best_g_per_mole = res.fun
                        best_ratios = res.x

        if best_ratios is None:
            return OptimizationResult3Phase(
                G_min=float('inf'), A_ratio_alpha=float('nan'), B_ratio_alpha=float('nan'),
                geometry_type=geometry_type, primary_phases=primary_phases,
                has_skin=has_skin, xB_skin=xB_skin_guess,
                r_vals=[]
            )

        # Unpack results
        A_ratio_at_min, B_ratio_at_min = best_ratios[0], best_ratios[1]
        xB_skin_at_min = best_ratios[2] if has_skin else xB_skin_guess
        skin_val_at_min = xB_skin_at_min if has_skin else None

        # Final detailed calculation
        phases, skin = self.calculator._update_phases_based_on_skin(primary_phases, skin_val_at_min)
        T_dep = self.calculator._get_T_dependent_vars(T, phases)
        n_mp, x_mp, r_vals = self.calculator._calc_mole_splits_and_geo(
            A_ratio_at_min, B_ratio_at_min, n_total, xB_total, phases, geometry_type, skin, T, T_dep
        )

        return OptimizationResult3Phase(
            G_min=best_g_per_mole * n_total,
            A_ratio_alpha=A_ratio_at_min, B_ratio_alpha=B_ratio_at_min,
            geometry_type=geometry_type, primary_phases=primary_phases,
            has_skin=has_skin, xB_skin=xB_skin_at_min,
            n_alpha=np.sum(n_mp[:, 0]), n_beta=np.sum(n_mp[:, 1]),
            xB_alpha=x_mp[1, 0], xB_beta=x_mp[1, 1],
            r_vals=r_vals.tolist()
        )
