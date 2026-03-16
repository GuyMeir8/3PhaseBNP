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

def process_temperature_series_task(task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Worker function to process a specific configuration across a temperature series.
    Utilizes warm-starting to pass the solution from T_i to T_{i+1}.
    """
    temperatures = task_data['temperatures']
    xB_total = task_data['xB_total']
    n_total = task_data['n_total']
    config = task_data['config']
    task_type = task_data['task_type']
    
    results_list = []
    
    try:
        if task_type == "SinglePhase":
            phase = task_data['phase']
            for T in temperatures:
                try:
                    G_single, r_single = _calculate_single_phase_energy(config, T, n_total, xB_total, phase)
                    
                    results_list.append({
                        "T": T, "xB_total": xB_total, "n_total": n_total,
                        "G_min": G_single, "Geometry": "SinglePhase",
                        "PhaseAlpha": phase, "PhaseBeta": "None",
                        "HasSkin": False, "xB_skin": np.nan,
                        "A_ratio_alpha": 1.0, "B_ratio_alpha": 1.0,
                        "n_alpha": n_total, "n_beta": 0.0,
                        "xB_alpha": xB_total, "xB_beta": np.nan,
                        "r_1": r_single, "r_2": np.nan, "r_3": np.nan, "r_4": np.nan
                    })
                except Exception:
                    pass

        elif task_type == "MultiPhase":
            optimizer = BNPOptimizer3Phase(config)
            geo = task_data['geometry']
            phases = task_data['phases']
            has_skin = task_data['has_skin']
            
            current_guess = None
            
            for T in temperatures:
                try:
                    # Full search only on the first step or if we lost the trail
                    needs_exhaustive = (current_guess is None)
                    
                    res: OptimizationResult3Phase = optimizer.find_minimum_energy(
                        T=T,
                        n_total=n_total,
                        xB_total=xB_total,
                        primary_phases=phases,
                        geometry_type=geo,
                        has_skin=has_skin,
                        xB_skin_guess=current_guess[2] if (has_skin and current_guess and len(current_guess) > 2) else 0.5,
                        initial_guess=current_guess,
                        exhaustive_search=needs_exhaustive
                    )
                    
                    r_list = res.r_vals if res.r_vals is not None else []
                    r_pad = r_list + [np.nan] * (4 - len(r_list))

                    results_list.append({
                        "T": T, "xB_total": xB_total, "n_total": n_total,
                        "G_min": res.G_min, "Geometry": geo,
                        "PhaseAlpha": phases[0], "PhaseBeta": phases[1],
                        "HasSkin": has_skin,
                        "xB_skin": res.xB_skin if has_skin else np.nan,
                        "A_ratio_alpha": res.A_ratio_alpha,
                        "B_ratio_alpha": res.B_ratio_alpha,
                        "n_alpha": res.n_alpha, "n_beta": res.n_beta,
                        "xB_alpha": res.xB_alpha, "xB_beta": res.xB_beta,
                        "r_1": r_pad[0], "r_2": r_pad[1], "r_3": r_pad[2], "r_4": r_pad[3]
                    })
                    
                    # Update guess for the next temperature step
                    if res.G_min != float('inf') and res.G_min < 1.0 and not np.isnan(res.A_ratio_alpha):
                        if has_skin:
                            current_guess = [res.A_ratio_alpha, res.B_ratio_alpha, res.xB_skin]
                        else:
                            current_guess = [res.A_ratio_alpha, res.B_ratio_alpha]
                    else:
                        current_guess = None
                        
                except Exception:
                    current_guess = None

    except Exception:
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

        temperatures = self.config.temperature_values

        for xB in self.config.xb_values:
            
            # 1. Single Phase Tasks
            for phase in self.config.phases:
                tasks.append({
                    'task_type': 'SinglePhase',
                    'temperatures': temperatures, 'xB_total': xB, 'n_total': n_total, 'config': self.config,
                    'phase': phase
                })

            # 2. Multi Phase Tasks
            for geo in geometries:
                for phases in phase_pairs:
                    for has_skin in skin_options:
                        
                        # Janus Symmetry: Skip redundant pairs
                        if geo == "Janus" and phases[0] > phases[1]:
                            continue

                        # Macroscopic (n=1) constraints
                        if abs(n_total - 1.0) < 1e-9:
                            if has_skin: continue
                            if geo != "Core Shell": continue
                            if phases == ("Liquid", "Liquid"): continue

                        tasks.append({
                            'task_type': 'MultiPhase',
                            'temperatures': temperatures, 'xB_total': xB, 'n_total': n_total, 'config': self.config,
                            'geometry': geo,
                            'phases': phases,
                            'has_skin': has_skin
                        })
        return tasks

    def get_suspect_points(self, df: pd.DataFrame) -> List[Tuple[float, float]]:
        """Identifies isolated anomalous points in the grid that may have failed to converge."""
        df_valid = df[(df["G_min"] < 1.0) & (~np.isinf(df["G_min"]))].copy()
        if df_valid.empty:
            return []
            
        try:
            # Find the absolute min energy config per point to build the map
            idx = df_valid.groupby(["T", "xB_total"])["G_min"].idxmin()
            df_min = df_valid.loc[idx].copy()
            
            def make_simple_label(row):
                return f"{row['Geometry']}_{row['PhaseAlpha']}_{row['PhaseBeta']}_{row['HasSkin']}"
                
            df_min["label"] = df_min.apply(make_simple_label, axis=1)
            
            grid_label_df = df_min.pivot(index="T", columns="xB_total", values="label")
            grid_label = grid_label_df.values
            
            grid_xb_df = df_min.pivot(index="T", columns="xB_total", values="xB_alpha")
            grid_xb = grid_xb_df.values
            
            rows, cols = grid_label.shape
            T_vals = grid_label_df.index.values
            xB_vals = grid_label_df.columns.values
            
            suspects = []
            for r in range(rows):
                for c in range(cols):
                    val_label = grid_label[r, c]
                    if pd.isna(val_label): continue
                    
                    matching_neighbors = 0
                    valid_neighbors = 0
                    
                    val_xb = grid_xb[r, c]
                    neighbor_xbs = []
                    
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                n_val_label = grid_label[nr, nc]
                                if pd.notna(n_val_label):
                                    valid_neighbors += 1
                                    if n_val_label == val_label:
                                        matching_neighbors += 1
                                        
                                        # ONLY append xB if it's the exact same phase geometry
                                        n_val_xb = grid_xb[nr, nc]
                                        if pd.notna(n_val_xb):
                                            neighbor_xbs.append(n_val_xb)
                                        
                    is_edge = (r == 0) or (r == rows - 1) or (c == 0) or (c == cols - 1)
                    limit = 0 if is_edge else 1
                    
                    is_suspect = False
                    
                    # Condition 1: Label completely disagrees with neighbors
                    if valid_neighbors > 0 and matching_neighbors <= limit:
                        is_suspect = True
                    # Condition 2: Label matches, but the phase composition violently inverted
                    elif len(neighbor_xbs) > 0:
                        avg_xb = sum(neighbor_xbs) / len(neighbor_xbs)
                        if abs(val_xb - avg_xb) > 0.4:
                            is_suspect = True
                            
                    if is_suspect:
                        suspects.append((T_vals[r], xB_vals[c]))
                        
            return suspects
        except Exception as e:
            print(f"Warning: Inspector encountered an issue during speckle detection: {e}")
            return []

    def generate_patch_tasks_for_n(self, n_total: float, suspect_points: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Generates exhaustive search tasks specifically for suspect points."""
        tasks = []
        geometries = self.config.geometries
        phase_pairs = list(itertools.product(self.config.phases, repeat=2))
        skin_options = [False, True]

        for T_susp, xB_susp in suspect_points:
            for geo in geometries:
                for phases in phase_pairs:
                    for has_skin in skin_options:
                        if geo == "Janus" and phases[0] > phases[1]: continue
                        if abs(n_total - 1.0) < 1e-9:
                            if has_skin: continue
                            if geo != "Core Shell": continue
                            if phases == ("Liquid", "Liquid"): continue

                        tasks.append({
                            'task_type': 'MultiPhase',
                            'temperatures': [T_susp], # A single element list triggers a heavy exhaustive search (no warm-start)
                            'xB_total': xB_susp,
                            'n_total': n_total,
                            'config': self.config,
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
                delayed(process_temperature_series_task)(task) for task in tasks
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

            df = pd.DataFrame(flat_results)
            
            # --- AUTO-FIX / INSPECTOR ---
            suspect_points = self.get_suspect_points(df)
            if suspect_points:
                print(f"Inspector found {len(suspect_points)} suspect points. Queueing deep patch search...")
                patch_tasks = self.generate_patch_tasks_for_n(n_total, suspect_points)
                
                nested_patch_results = Parallel(n_jobs=n_jobs, verbose=5)(
                    delayed(process_temperature_series_task)(task) for task in patch_tasks
                )
                
                flat_patch = [item for sublist in nested_patch_results for item in sublist]
                if flat_patch:
                    df = pd.concat([df, pd.DataFrame(flat_patch)], ignore_index=True)
                    # Deduplicate to keep only the absolute lowest G_min if multiple calculations exist for the same config
                    df = df.sort_values(by=["T", "xB_total", "Geometry", "PhaseAlpha", "PhaseBeta", "HasSkin", "G_min"])
                    df = df.drop_duplicates(subset=["T", "xB_total", "Geometry", "PhaseAlpha", "PhaseBeta", "HasSkin"], keep='first')
            # ---------------------------

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