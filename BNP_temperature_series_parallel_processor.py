import pandas as pd
import numpy as np
import time
import datetime
import os
from typing import Tuple, Dict, Any, List
import itertools
from joblib import Parallel, delayed

from configurations_3_phase import ThreePhaseConfiguration, low_res_configuration, standard_configuration
from BNP_optimizer_3_phase import BNPOptimizer3Phase, OptimizationResult3Phase
from BNP_Gibbs_en_calc_3_phase import GibbsEnergyCalculator3Phase

def _calculate_single_phase_energy(
    config: ThreePhaseConfiguration,
    T: float,
    n_total: float,
    xB_total: float,
    phase: str
) -> Tuple[float, float]:
    """
    Helper to calculate single phase energy (Ideal + Excess + Surface).
    Assumes a spherical droplet of the given phase.
    """
    calc = GibbsEnergyCalculator3Phase(config)
    
    # 1. Setup Moles and Fractions
    n_A = n_total * (1 - xB_total)
    n_B = n_total * xB_total
    
    # Shape: (Materials=2, Phases=1)
    n_mp = np.array([[n_A], [n_B]])
    x_mp = np.array([[1 - xB_total], [xB_total]])
    phases = (phase,)
    
    # Get Temperature Dependent Variables
    T_dep = calc._get_T_dependent_vars(T, phases)

    # 2. Calculate Bulk Energies
    G_ideal = calc._calc_G_ideal(n_mp, x_mp, T, phases, T_dep)
    G_excess = calc._calc_G_excess(n_mp, x_mp, T, phases, T_dep)
    
    # 3. Calculate Surface Energy (Sphere)
    v_mp = T_dep.v_mp # shape (2, 1)
    V_total = n_A * v_mp[0,0] + n_B * v_mp[1,0]
    
    # Use the static method for radius
    r = calc.calc_r_from_V(V_total)
    area = 4 * np.pi * r**2
    
    # Calculate Surface Tension (Phase -> Vacuum)
    # The calculator's internal method solves for surface enrichment
    sigma = calc._calculate_surface_tension(
        xB_alpha=xB_total,
        xB_beta=None,
        phase_alpha=phase,
        phase_beta=None,
        T=T,
        phases=phases,
        T_dependent_parameters=T_dep
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
    
    results_list = []
    
    try:
        if task_type == "SinglePhase":
            phase = task_data['phase']
            G_single, r_single = _calculate_single_phase_energy(config, T, n_total, xB_total, phase)
            
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
                "r_1": r_single,
                "r_2": np.nan,
                "r_3": np.nan,
                "r_4": np.nan
            })

        elif task_type == "MultiPhase":
            # Initialize optimizer with config directly
            optimizer = BNPOptimizer3Phase(config)
            
            geo = task_data['geometry']
            phases = task_data['phases']
            has_skin = task_data['has_skin']
            
            # Use exhaustive search for robustness in series processing
            res: OptimizationResult3Phase = optimizer.find_minimum_energy(
                T=T,
                n_total=n_total,
                xB_total=xB_total,
                primary_phases=phases,
                geometry_type=geo,
                has_skin=has_skin,
                xB_skin_guess=0.5, # Default guess
                exhaustive_search=True
            )
            
            r_list = res.r_vals if res.r_vals is not None else []
            r_pad = r_list + [np.nan] * (4 - len(r_list))

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
                "r_1": r_pad[0],
                "r_2": r_pad[1],
                "r_3": r_pad[2],
                "r_4": r_pad[3]
            })
    except Exception:
        # Return empty list on failure so the main loop continues
        pass

    return results_list

class BNPSeriesProcessor:
    def __init__(self, config: ThreePhaseConfiguration):
        self.config = config
    
    def generate_tasks_for_n(self, n_total: float) -> List[Dict[str, Any]]:
        """Generates a flat list of all specific tasks to run for a given n_total."""
        tasks = []
        
        geometries = self.config.geometries
        phase_pairs = list(itertools.product(self.config.phases, repeat=2))
        skin_options = [False, True]

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
                        for has_skin in skin_options:
                            
                            # Janus Symmetry: Skip redundant pairs (e.g. Liquid-FCC if FCC-Liquid done)
                            if geo == "Janus" and phases[0] > phases[1]:
                                continue

                            # Macroscopic (n=1) constraints (Only Core Shell, No Skin for n=1)
                            if abs(n_total - 1.0) < 1e-9:
                                if has_skin: continue
                                if geo != "Core Shell": continue
                                if phases == ("Liquid", "Liquid"): continue

                            tasks.append({
                                'task_type': 'MultiPhase',
                                'T': T, 'xB_total': xB, 'n_total': n_total, 'config': self.config,
                                'geometry': geo,
                                'phases': phases,
                                'has_skin': has_skin
                            })
        return tasks

    def run(self, n_jobs: int = -1, auto_show: bool = True):
        """
        Runs the parallel processing over the configuration grid.
        Automatically saves results to Results/ folder and opens the generated plots.
        """
        print(f"--- Starting Simulation ---")
        print(f"Config: {self.config.base_file_name}")
        print(f"Jobs (Cores): {n_jobs if n_jobs != -1 else 'All Available'}")
        
        # Create Results directory if it doesn't exist
        output_dir = "Results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        chunk_files = []

        for n_total in self.config.n_total_values:
            print(f"\nProcessing n_total = {n_total}")
            tasks = self.generate_tasks_for_n(n_total)
            print(f"Points to process: {len(tasks)}")
            
            start_time = time.time()
            
            # Run Parallel Processing
            # n_jobs=-1 uses all available cores. verbose=5 shows progress.
            nested_results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(process_single_task)(task) for task in tasks
            )
                
            # Flatten the list of lists into a single list of dictionaries
            flat_results = [item for sublist in nested_results for item in sublist]
                
            end_time = time.time()
            duration = end_time - start_time
            duration_formatted = str(datetime.timedelta(seconds=duration))
            print(f"Chunk completed in {duration_formatted}.")
            
            if not flat_results:
                print(f"No results generated for n_total={n_total}.")
                continue

            # Save Chunk Results
            df = pd.DataFrame(flat_results)
            df = df.sort_values(by=["T", "xB_total", "G_min"])
            df["Geometry"] = df["Geometry"].replace({"Core Shell": "Core_Shell"})
            
            chunk_filename = f"{self.config.base_file_name}_n_{n_total}_{timestamp}.csv"
            chunk_filepath = os.path.join(output_dir, chunk_filename)
            df.to_csv(chunk_filepath, index=False)
            chunk_files.append(chunk_filepath)
            print(f"Chunk results saved to {chunk_filepath}")

        if not chunk_files:
            print("\nNo valid results were generated across all n_total values.")
            return

        print("\nCombining chunks into master file...")
        combined_df = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
        master_filename = f"{self.config.base_file_name}_{timestamp}.csv"
        master_filepath = os.path.join(output_dir, master_filename)
        combined_df.to_csv(master_filepath, index=False)
        print(f"Master file saved to {master_filepath}")

        # Automatically Open and Display Plots
        print("Generating and displaying phase diagrams...")
        try:
            from plotting_3_phase import PhaseDiagramPlotting3Phase
            PhaseDiagramPlotting3Phase(master_filepath, save_dir=output_dir, timestamp=timestamp, auto_show=auto_show)
        except Exception as e:
            print(f"Warning: Could not open plots. Error: {e}")
        
if __name__ == "__main__":
    # Select Configuration
    # config = standard_configuration
    config = low_res_configuration
    
    processor = BNPSeriesProcessor(config)
    processor.run()