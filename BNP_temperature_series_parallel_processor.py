import pandas as pd
import numpy as np
import time
import datetime
import os
from typing import Tuple, Dict, Any, List
import itertools
from joblib import Parallel, delayed

from configurations_3_phase import ThreePhaseConfiguration, low_res_configuration, standard_configuration
from system_data import SystemData
from BNP_optimizer_3_phase import BNPOptimizer3Phase, OptimizationResult3Phase
from BNP_Gibbs_en_calc_3_phase import GibbsEnergyCalculator3Phase

def _calculate_single_phase_energy(
    system_data: SystemData,
    T: float,
    n_total: float,
    xB_total: float,
    phase: str
) -> Tuple[float, float]:
    """
    Helper to calculate single phase energy (Ideal + Excess + Surface).
    Assumes a spherical droplet of the given phase.
    """
    calc = GibbsEnergyCalculator3Phase(system_data)
    
    # 1. Setup Moles and Fractions
    n_A = n_total * (1 - xB_total)
    n_B = n_total * xB_total
    
    n_mp = np.array([[n_A], [n_B]])
    x_mp = np.array([[1 - xB_total], [xB_total]])
    phases = (phase,)
    
    # 2. Calculate Bulk Energies
    G_ideal = calc.calc_G_ideal(n_mp, x_mp, T, phases)
    G_excess = calc.calc_G_excess(n_mp, x_mp, T, phases)
    
    # 3. Calculate Surface Energy (Sphere)
    v_mp = calc._get_v_mp(T, phases) # shape (2, 1)
    V_total = n_A * v_mp[0,0] + n_B * v_mp[1,0]
    r = calc.calc_r_from_V(V_total)
    area = 4 * np.pi * r**2
    
    # Calculate Sigma (Phase -> Vacuum)
    sigma = calc._calc_sigma(
        curr_phases=phases,
        T=T,
        v_gamma=v_mp[:,0],
        x_gamma=x_mp[:,0]
    )
    
    G_surface = area * sigma
    
    return G_ideal + G_excess + G_surface, r

def process_single_task(task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Worker function to process a specific calculation task.
    task_data contains: T, xB_total, n_total, config, task_type, and specific params.
    """
    T = task_data['T']
    xB_total = task_data['xB_total']
    n_total = task_data['n_total']
    config = task_data['config']
    task_type = task_data['task_type']
    
    # Re-initialize system data for this process to ensure thread safety
    system_data = SystemData(config)
    
    results_list = []
    
    try:
        if task_type == "SinglePhase":
            phase = task_data['phase']
            G_single, r_single = _calculate_single_phase_energy(system_data, T, n_total, xB_total, phase)
            
            results_list.append({
                "T": T,
                "xB_total": xB_total,
                "n_total": n_total,
                "G_min": G_single,
                "Geometry": "SinglePhase",
                "PhaseAlpha": phase,
                "PhaseBeta": "None",
                "HasSkin": False,
                "xB_skin": np.nan,
                "A_ratio_alpha": 1.0,
                "B_ratio_alpha": 1.0,
                "n_alpha": n_total,
                "n_beta": 0.0,
                "xB_alpha": xB_total,
                "xB_beta": np.nan,
                "r_vals": [r_single]
            })

        elif task_type == "MultiPhase":
            optimizer = BNPOptimizer3Phase(system_data)
            geo = task_data['geometry']
            phases = task_data['phases']
            has_skin = task_data['has_skin']
            
            res: OptimizationResult3Phase = optimizer.find_minimum_energy(
                T=T,
                n_total=n_total,
                xB_total=xB_total,
                primary_phases=phases,
                geometry_type=geo,
                has_skin=has_skin,
                xB_skin_guess=0.2
            )
            
            results_list.append({
                "T": T,
                "xB_total": xB_total,
                "n_total": n_total,
                "G_min": res.G_min,
                "Geometry": geo,
                "PhaseAlpha": phases[0],
                "PhaseBeta": phases[1],
                "HasSkin": has_skin,
                "xB_skin": res.xB_skin if has_skin else np.nan,
                "A_ratio_alpha": res.A_ratio_alpha,
                "B_ratio_alpha": res.B_ratio_alpha,
                "n_alpha": res.n_alpha,
                "n_beta": res.n_beta,
                "xB_alpha": res.xB_alpha,
                "xB_beta": res.xB_beta,
                "r_vals": res.r_vals
            })
    except Exception:
        # Return empty list on failure so the main loop continues
        pass

    return results_list

class BNPSeriesProcessor:
    def __init__(self, config: ThreePhaseConfiguration):
        self.config = config
    
    def generate_tasks(self) -> List[Dict[str, Any]]:
        """Generates a flat list of all specific tasks to run."""
        tasks = []
        
        geometries = self.config.geometries
        phase_pairs = list(itertools.product(self.config.phases, repeat=2))
        skin_options = [False, True]

        for n_total in self.config.n_total_values:
            for T in self.config.temperature_values:
                for xB in self.config.xb_values:
                    
                    # 1. Single Phase Tasks
                    for phase in self.config.phases:
                        tasks.append({
                            'task_type': 'SinglePhase',
                            'T': T, 'xB_total': xB, 'n_total': n_total, 'config': self.config,
                            'phase': phase
                        })

                    # 2. Multi Phase Tasks
                    for geo in geometries:
                        for phases in phase_pairs:
                            # Skip Liquid-Liquid
                            if phases[0] == "Liquid" and phases[1] == "Liquid":
                                continue

                            for has_skin in skin_options:
                                
                                # Janus Symmetry: Skip redundant pairs (e.g. Liquid-FCC if FCC-Liquid done)
                                if geo == "Janus" and phases[0] > phases[1]:
                                    continue

                                # Macroscopic (n=1) constraints
                                if abs(n_total - 1.0) < 1e-9:
                                    if has_skin: continue
                                    if geo != "Janus": continue

                                tasks.append({
                                    'task_type': 'MultiPhase',
                                    'T': T, 'xB_total': xB, 'n_total': n_total, 'config': self.config,
                                    'geometry': geo,
                                    'phases': phases,
                                    'has_skin': has_skin
                                })
        return tasks

    def run(self, n_jobs: int = -1):
        """
        Runs the parallel processing over the configuration grid.
        """
        # 1. Generate Tasks
        print("Generating tasks...")
        tasks = self.generate_tasks()
        
        print(f"--- Starting Simulation ---")
        print(f"Config: {self.config.base_file_name}")
        print(f"Points to process: {len(tasks)}")
        print(f"Jobs (Cores): {n_jobs if n_jobs != -1 else 'All Available'}")
        
        start_time = time.time()
        
        # 2. Run Parallel Processing
        # n_jobs=-1 uses all available cores. verbose=5 shows progress.
        nested_results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(process_single_task)(task) for task in tasks
        )
            
        # Flatten the list of lists into a single list of dictionaries
        flat_results = [item for sublist in nested_results for item in sublist]
            
        end_time = time.time()
        duration = end_time - start_time
        duration_formatted = str(datetime.timedelta(seconds=duration))
        print(f"Simulation completed in {duration_formatted}.")
        
        # 3. Save Results
        df = pd.DataFrame(flat_results)
        
        # Sort for readability
        df = df.sort_values(by=["n_total", "T", "xB_total", "G_min"])
        
        # Create Results directory if it doesn't exist
        output_dir = "Results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.base_file_name}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")

if __name__ == "__main__":
    # Select Configuration
    # config = standard_configuration
    config = low_res_configuration
    
    processor = BNPSeriesProcessor(config)
    processor.run()
