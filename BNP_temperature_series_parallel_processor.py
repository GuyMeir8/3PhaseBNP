import pandas as pd
import numpy as np
import time
import datetime
import os
from typing import Tuple, Dict, Any, List, Optional
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
            
            task_initial_guess = task_data.get('initial_guess')
            current_guess = task_initial_guess
            is_first_step = True
            
            for T in temperatures:
                try:
                    # Full search only on the first step or if we lost the trail
                    needs_exhaustive = (current_guess is None) or (is_first_step and task_initial_guess is None)
                    is_first_step = False
                    
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
                            if phases[0] > phases[1]: continue

                        tasks.append({
                            'task_type': 'MultiPhase',
                            'temperatures': temperatures, 'xB_total': xB, 'n_total': n_total, 'config': self.config,
                            'geometry': geo,
                            'phases': phases,
                            'has_skin': has_skin
                        })
        return tasks

    def get_suspect_points(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identifies isolated anomalous points in the grid that may have failed to converge."""
        df_valid = df[(df["G_min"] < 1.0) & (~np.isinf(df["G_min"]))].copy()
        if df_valid.empty:
            return []
            
        try:
            # Find the absolute min energy config per point to build the map
            idx = df_valid.groupby(["T", "xB_total"])["G_min"].idxmin()
            df_min = df_valid.loc[idx].copy()
            
            def make_comparison_label(row):
                geo = row["Geometry"]
                pa = str(row["PhaseAlpha"]).split("(")[0]
                pb = str(row["PhaseBeta"]).split("(")[0]
                skin = "_Skin" if row["HasSkin"] else "_NoSkin"
                return f"{geo}_{pa}_{pb}{skin}"
                
            df_min["comparison_label"] = df_min.apply(make_comparison_label, axis=1)
            df_min["phase_fraction"] = df_min["n_alpha"] / df_min["n_total"]
            # Pivot all columns needed for comparison, configs, and initial guesses
            pivot_cols = ["comparison_label", "xB_alpha", "xB_beta", "phase_fraction", "Geometry", "PhaseAlpha", "PhaseBeta", "HasSkin", "A_ratio_alpha", "B_ratio_alpha", "xB_skin", "G_min"]
            grids = {}
            for col in pivot_cols:
                try:
                    grids[col] = df_min.pivot(index="T", columns="xB_total", values=col)
                except ValueError:
                    grids[col] = df_min.pivot_table(index="T", columns="xB_total", values=col, aggfunc='first')

            grid_label = grids["comparison_label"].values
            grid_xba = grids["xB_alpha"].values
            grid_xbb = grids["xB_beta"].values
            grid_frac = grids["phase_fraction"].values
            grid_geo = grids["Geometry"].values
            grid_pa = grids["PhaseAlpha"].values
            grid_pb = grids["PhaseBeta"].values
            grid_hs = grids["HasSkin"].values
            grid_ara = grids["A_ratio_alpha"].values
            grid_bra = grids["B_ratio_alpha"].values
            grid_xbs_val = grids["xB_skin"].values
            grid_gmin = grids["G_min"].values
            
            rows, cols = grid_label.shape
            T_vals = grids["comparison_label"].index.values
            xB_vals = grids["comparison_label"].columns.values
            
            suspects = []
            for r in range(rows):
                for c in range(cols):
                    val_label = grid_label[r, c]
                    if pd.isna(val_label): continue
                    
                    neighbor_comparison_vals = []
                    matching_neighbor_xba = []
                    matching_neighbor_xbb = []
                    matching_neighbor_frac = []

                    val_xba = grid_xba[r, c]
                    val_xbb = grid_xbb[r, c]
                    val_frac = grid_frac[r, c]
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                n_val_label = grid_label[nr, nc]
                                if pd.notna(n_val_label):
                                    neighbor_comparison_vals.append(n_val_label)
                                    if n_val_label == val_label:
                                        n_val_xba = grid_xba[nr, nc]
                                        if pd.notna(n_val_xba):
                                            matching_neighbor_xba.append(n_val_xba)
                                        n_val_xbb = grid_xbb[nr, nc]
                                        if pd.notna(n_val_xbb):
                                            matching_neighbor_xbb.append(n_val_xbb)
                                        n_val_frac = grid_frac[nr, nc]
                                        if pd.notna(n_val_frac):
                                            matching_neighbor_frac.append(n_val_frac)


                    if not neighbor_comparison_vals: continue
                        
                    valid_neighbors = len(neighbor_comparison_vals)
                    neighbors_matching_self = neighbor_comparison_vals.count(val_label)
                    
                    is_edge = (r == 0) or (r == rows - 1) or (c == 0) or (c == cols - 1)
                    is_suspect = False
                    
                    # Condition 1: Label completely disagrees with neighbors
                    if is_edge:
                        if valid_neighbors > 0 and neighbors_matching_self <= int(0.4 * valid_neighbors):
                            is_suspect = True
                    else:
                        if neighbors_matching_self <= 4:
                            is_suspect = True
                    
                    # Condition 2: Label matches, but the phase composition jumped
                    if not is_suspect:
                        for n_xba in matching_neighbor_xba:
                            if pd.notna(val_xba) and abs(val_xba - n_xba) > 0.2:
                                is_suspect = True
                                break
                        if not is_suspect:
                            for n_xbb in matching_neighbor_xbb:
                                if pd.notna(val_xbb) and abs(val_xbb - n_xbb) > 0.2:
                                    is_suspect = True
                                    break
                    
                    if not is_suspect:
                            for n_frac in matching_neighbor_frac:
                                if pd.notna(val_frac) and abs(val_frac - n_frac) > 0.15:
                                    is_suspect = True
                                    break
                                
                    if is_suspect:
                        # Get current config
                        current_config = {
                            'geometry': grid_geo[r, c],
                            'phases': (grid_pa[r, c], grid_pb[r, c]),
                            'has_skin': bool(grid_hs[r, c])
                        }

                        # Find majority neighbor config
                        from collections import Counter
                        counts = Counter(neighbor_comparison_vals)
                        most_common_comparison_label, _ = counts.most_common(1)[0]
                        
                        best_majority_neighbor_g = float('inf')
                        best_majority_neighbor_coords = None
                        
                        # Find the best instance (lowest G) of the majority neighbor
                        for dr_find in [-1, 0, 1]:
                            for dc_find in [-1, 0, 1]:
                                if dr_find == 0 and dc_find == 0: continue
                                nr_find, nc_find = r + dr_find, c + dc_find
                                if 0 <= nr_find < rows and 0 <= nc_find < cols:
                                    if grids["comparison_label"].values[nr_find, nc_find] == most_common_comparison_label:
                                        neighbor_g = grid_gmin[nr_find, nc_find]
                                        if pd.notna(neighbor_g) and neighbor_g < best_majority_neighbor_g:
                                            best_majority_neighbor_g = neighbor_g
                                            best_majority_neighbor_coords = (nr_find, nc_find)

                        if best_majority_neighbor_coords:
                            nr_best, nc_best = best_majority_neighbor_coords

                            majority_config = {
                                'geometry': grid_geo[nr_best, nc_best],
                                'phases': (grid_pa[nr_best, nc_best], grid_pb[nr_best, nc_best]),
                                'has_skin': bool(grid_hs[nr_best, nc_best])
                            }

                            majority_guess = [
                                grid_ara[nr_best, nc_best],
                                grid_bra[nr_best, nc_best]
                            ]
                            
                            if majority_config['has_skin']:
                                xbs_val = grid_xbs_val[nr_best, nc_best]
                                if pd.notna(xbs_val):
                                    majority_guess.append(xbs_val)
                                
                            suspects.append({'T': T_vals[r], 'xB_total': xB_vals[c],
                                             'current_config': current_config, 'majority_config': majority_config,
                                             'majority_guess': majority_guess})
                        
            return suspects
        except Exception as e:
            print(f"Warning: Inspector encountered an issue during speckle detection: {e}")
            return []

    def generate_patch_tasks_for_n(self, n_total: float, points: List[Tuple[float, float]], filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generates exhaustive search tasks specifically for a list of points, with optional filters."""
        if filters is None:
            filters = {}

        tasks = []
        
        # Determine which configurations to generate based on filters. If no filters, generate all.
        gen_geometries = filters.get('geometries', list(self.config.geometries) + ["SinglePhase"])
        gen_phase_pairs = filters.get('phase_pairs', list(itertools.product(self.config.phases, repeat=2)))
        gen_skin_options = filters.get('skin_options', [False, True])
        gen_single_phases = filters.get('single_phases', self.config.phases)

        for T_point, xB_point in points:
            # 1. Single Phase Tasks
            if "SinglePhase" in gen_geometries:
                for phase in self.config.phases:
                    if phase in gen_single_phases:
                        tasks.append({
                            'task_type': 'SinglePhase',
                            'temperatures': [T_point],
                            'xB_total': xB_point,
                            'n_total': n_total,
                            'config': self.config,
                            'phase': phase
                        })

            # 2. Multi Phase Tasks
            for geo in self.config.geometries:
                if geo not in gen_geometries:
                    continue
                for phases_tuple in list(itertools.product(self.config.phases, repeat=2)):
                    phases = tuple(phases_tuple)
                    # Check if the tuple is in the list of allowed pairs
                    if not any(tuple(p) == phases for p in gen_phase_pairs):
                        continue
                    for has_skin in [False, True]:
                        if has_skin not in gen_skin_options:
                            continue
                            
                        # Existing rules for valid configurations
                        if geo == "Janus" and phases[0] > phases[1]: continue
                        if abs(n_total - 1.0) < 1e-9:
                            if has_skin: continue
                            if geo != "Core Shell": continue
                            if phases == ("Liquid", "Liquid"): continue
                            if phases[0] > phases[1]: continue
                        
                        tasks.append({
                            'task_type': 'MultiPhase',
                            'temperatures': [T_point],
                            'xB_total': xB_point,
                            'n_total': n_total,
                            'config': self.config,
                            'geometry': geo,
                            'phases': phases,
                            'has_skin': has_skin
                        })
        return tasks

    def generate_autofix_tasks(self, n_total: float, suspect_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generates targeted recalculation tasks for suspect points."""
        tasks = []
        seen_tasks = set()

        def create_task(T, xB, config_dict, initial_guess=None):
            geo = config_dict['geometry']
            phases = config_dict['phases']
            has_skin = config_dict['has_skin']

            if geo == 'SinglePhase':
                task = {
                    'task_type': 'SinglePhase', 'temperatures': [T], 'xB_total': xB,
                    'n_total': n_total, 'config': self.config, 'phase': phases[0]
                }
                task_id = ('SinglePhase', T, xB, n_total, phases[0])
            else:
                task = {
                    'task_type': 'MultiPhase', 'temperatures': [T], 'xB_total': xB,
                    'n_total': n_total, 'config': self.config, 'geometry': geo,
                    'phases': phases, 'has_skin': has_skin
                }
                if initial_guess is not None:
                    task['initial_guess'] = initial_guess
                task_id = ('MultiPhase', T, xB, n_total, geo, phases[0], phases[1], has_skin)
            
            return task, task_id

        for suspect in suspect_details:
            T = suspect['T']
            xB = suspect['xB_total']
            maj_guess = suspect.get('majority_guess')

            # Task 1: Recalculate the current (suspect) configuration
            task1, id1 = create_task(T, xB, suspect['current_config'], maj_guess)
            if id1 not in seen_tasks:
                tasks.append(task1)
                seen_tasks.add(id1)

            # Task 2: Recalculate the majority neighbor's configuration
            if suspect['majority_config'] != suspect['current_config']:
                task2, id2 = create_task(T, xB, suspect['majority_config'], maj_guess)
                if id2 not in seen_tasks:
                    tasks.append(task2)
                    seen_tasks.add(id2)
        
        return tasks

    def run_patch_on_file(self, # type: ignore
                      filepath: str,
                      areas_to_patch: List[Dict[str, Any]],
                      n_jobs: int = -1,
                      auto_show: bool = True):
        """
        Runs an exhaustive search on specific areas for an existing results file and updates it.

        Args:
            filepath: Path to the CSV file to be patched.
            areas_to_patch: A list of dictionaries, each defining a rectangular area and
                            optional filters for calculations.
            n_jobs: Number of parallel jobs.
            auto_show: Whether to show the plot automatically after patching.
        """
        print(f"--- Starting Area Patch Run on {filepath} ---")
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            return

        try:
            df_original = pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return

        # Infer n_total from the file, as it's assumed to contain only one.
        n_total_values = df_original['n_total'].unique()
        if len(n_total_values) == 0:
            print("Error: No 'n_total' values found in the file.")
            return
        if len(n_total_values) > 1:
            print(f"Warning: Multiple n_total values found: {n_total_values}. Using the first one: {n_total_values[0]}")
        n_total = n_total_values[0]
        print(f"Inferred n_total = {n_total} from file.")

        all_patch_tasks = []
        all_points_to_recalculate = set()

        # Generate tasks for each defined area, applying specific filters.
        all_grid_points = df_original[['T', 'xB_total']].drop_duplicates().to_records(index=False)

        for area in areas_to_patch:
            t_min, t_max = area.get('T_range', (-np.inf, np.inf))
            xb_min, xb_max = area.get('xB_range', (-np.inf, np.inf))
            
            points_in_area = []
            for T, xB in all_grid_points: # Use a pre-queried list of unique points
                if t_min <= T <= t_max and xb_min <= xB <= xb_max:
                    points_in_area.append((T, xB))
            
            if not points_in_area:
                continue

            all_points_to_recalculate.update(points_in_area)
            
            filters = area.get('filters')
            
            patch_tasks_for_area = self.generate_patch_tasks_for_n(
                n_total, 
                points_in_area, 
                filters=filters
            )
            all_patch_tasks.extend(patch_tasks_for_area)

        # De-duplicate tasks. This is important if areas overlap.
        unique_tasks = []
        seen_tasks = set()
        for task in all_patch_tasks:
            task_id_parts = [
                task['task_type'], task['temperatures'][0], task['xB_total'], task['n_total'],
            ]
            if task['task_type'] == 'SinglePhase':
                task_id_parts.append(task['phase'])
            else:
                task_id_parts.extend([
                    task['geometry'], task['phases'][0], task['phases'][1], task['has_skin']
                ])
            
            task_id = tuple(task_id_parts)
            if task_id not in seen_tasks:
                unique_tasks.append(task)
                seen_tasks.add(task_id)
        
        patch_tasks = unique_tasks

        # Call the shared execution logic
        self._execute_patch_and_update_file(df_original, patch_tasks, filepath, n_jobs)

    def run_speckle_autofix_on_file(self,
                                  filepath: str,
                                  n_jobs: int = -1,
                                  auto_show: bool = True):
        """
        Loads a file, automatically detects speckles/anomalies, and runs an
        exhaustive recalculation on just those points, updating the file.
        """
        print(f"--- Starting Speckle Autofix Run on {filepath} ---")
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            return

        max_passes = 10
        previous_suspect_coords = set()

        for pass_num in range(1, max_passes + 1):
            try:
                df_original = pd.read_csv(filepath)
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                return

            # Infer n_total from the file
            n_total_values = df_original['n_total'].unique()
            if len(n_total_values) == 0:
                print("Error: No 'n_total' values found in the file.")
                return
            if len(n_total_values) > 1:
                print(f"Warning: Multiple n_total values found: {n_total_values}. Using the first one: {n_total_values[0]}")
            n_total = n_total_values[0]
            
            if pass_num == 1:
                print(f"Inferred n_total = {n_total} from file.")

            print(f"\n--- Autofix Pass {pass_num}/{max_passes} ---")

            # Automatically find suspect points
            suspect_details = self.get_suspect_points(df_original)
            if not suspect_details:
                print("Inspector found no suspect points to fix. Exiting loop.")
                break
            
            current_suspect_coords = set((s['T'], s['xB_total']) for s in suspect_details)
            if pass_num > 1 and current_suspect_coords == previous_suspect_coords:
                print("Remaining suspect points could not be improved further. Exiting loop.")
                break
            previous_suspect_coords = current_suspect_coords

            print(f"Inspector found {len(suspect_details)} suspect points. Queueing targeted deep search...")
            # Run a targeted search on only the suspect and majority-neighbor configs
            patch_tasks = self.generate_autofix_tasks(n_total, suspect_details)

            # Call the shared execution logic
            self._execute_patch_and_update_file(df_original, patch_tasks, filepath, n_jobs)
            
        else:
            print(f"\nReached maximum number of autofix passes ({max_passes}).")

    def _execute_patch_and_update_file(self,
                                     df_original: pd.DataFrame,
                                     patch_tasks: List[Dict[str, Any]],
                                     filepath: str,
                                     n_jobs: int = -1):
        """
        Private helper to execute a list of patch tasks with checkpointing,
        merge the results with an original dataframe, and save the updated file.
        """
        if not patch_tasks:
            print("No tasks to execute. No changes made.")
            return
            
        patch_checkpoint_filepath = filepath.replace(".csv", "_patch_checkpoint.csv")
        if os.path.exists(patch_checkpoint_filepath):
            os.remove(patch_checkpoint_filepath)
            print(f"Removed old patch checkpoint file: {patch_checkpoint_filepath}")
            
        print(f"Queueing {len(patch_tasks)} deep search tasks...")
        start_time = time.time()
        
        save_frequency = 100
        print(f"  Processing tasks continuously (saving checkpoint every {save_frequency} tasks)...")
        
        results_generator = Parallel(n_jobs=n_jobs, verbose=5, return_as="generator_unordered")(
            delayed(process_temperature_series_task)(task) for task in patch_tasks
        )
        
        tasks_completed = 0
        pending_results = []
        
        for task_result in results_generator:
            pending_results.extend(task_result)
            tasks_completed += 1
        
            if tasks_completed % save_frequency == 0:
                if pending_results:
                    df_batch = pd.DataFrame(pending_results)
                    header = not os.path.exists(patch_checkpoint_filepath)
                    df_batch.to_csv(patch_checkpoint_filepath, mode='a', header=header, index=False)
                    pending_results = []
                print(f"    [Patch Checkpoint] Safely wrote progress to disk (Task {tasks_completed})...")
        
        if pending_results:
            df_batch = pd.DataFrame(pending_results)
            header = not os.path.exists(patch_checkpoint_filepath)
            df_batch.to_csv(patch_checkpoint_filepath, mode='a', header=header, index=False)
        
        end_time = time.time()
        duration = str(datetime.timedelta(seconds=end_time - start_time))
        print(f"Recalculations completed in {duration}.")
        
        if not os.path.exists(patch_checkpoint_filepath):
            print("No valid patch results were generated.")
            print("\nNo changes made to the file.")
            return
        
        df_patch = pd.read_csv(patch_checkpoint_filepath)
        try:
            os.remove(patch_checkpoint_filepath)
        except Exception as e:
            print(f"Warning: Could not remove patch checkpoint file: {e}")
        
        df_combined = pd.concat([df_original, df_patch], ignore_index=True)
        
        # Standardize the naming before deduplication so old and new data match exactly
        df_combined["Geometry"] = df_combined["Geometry"].replace({"Core Shell": "Core_Shell"})
        
        # Define the unique key for a single calculation configuration
        config_cols = ["T", "xB_total", "n_total", "Geometry", "PhaseAlpha", "PhaseBeta", "HasSkin"]
    
        # Sort by G_min so that the best result for each configuration comes first
        df_combined = df_combined.sort_values(by=config_cols + ["G_min"])
    
        # Drop duplicates, keeping only the first occurrence (which has the lowest G_min)
        df_updated = df_combined.drop_duplicates(subset=config_cols, keep='first')
        
        print(f"Run complete. Total rows now: {len(df_updated)}")
        
        # Save the updated dataframe back to the original file path, creating a backup first.
        backup_filepath = filepath.replace(".csv", f"_backup_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
        os.rename(filepath, backup_filepath)
        print(f"\nOriginal file backed up to {backup_filepath}")
        
        df_updated.to_csv(filepath, index=False)
        print(f"Updated file saved to {filepath}")

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
            
            # --- Checkpoint / Resume Logic ---
            checkpoint_filepath = os.path.join(output_dir, f"{self.config.base_file_name}_n_{n_total}_checkpoint.csv")
            completed_task_ids = set()
            if os.path.exists(checkpoint_filepath):
                try:
                    df_cp = pd.read_csv(checkpoint_filepath)
                    if not df_cp.empty:
                        for _, row in df_cp[['xB_total', 'Geometry', 'PhaseAlpha', 'PhaseBeta', 'HasSkin']].drop_duplicates().iterrows():
                            xb_str = f"{float(row['xB_total']):.6g}"
                            if row['Geometry'] == 'SinglePhase':
                                completed_task_ids.add(f"SinglePhase_{xb_str}_{row['PhaseAlpha']}")
                            else:
                                completed_task_ids.add(f"MultiPhase_{xb_str}_{row['Geometry']}_{row['PhaseAlpha']}_{row['PhaseBeta']}_{bool(row['HasSkin'])}")
                        print(f"Resuming from checkpoint. Found {len(completed_task_ids)} completed task configurations.")
                except Exception as e:
                    print(f"Warning: Could not read checkpoint file {checkpoint_filepath}: {e}")
            
            def get_task_id(task):
                xb_str = f"{float(task['xB_total']):.6g}"
                if task['task_type'] == 'SinglePhase':
                    return f"SinglePhase_{xb_str}_{task['phase']}"
                else:
                    return f"MultiPhase_{xb_str}_{task['geometry']}_{task['phases'][0]}_{task['phases'][1]}_{bool(task['has_skin'])}"

            tasks_to_run = [t for t in tasks if get_task_id(t) not in completed_task_ids]
            print(f"Points to process: {len(tasks_to_run)} (Skipped {len(tasks) - len(tasks_to_run)})")
            
            start_time = time.time()
            
            # --- Continuous Parallel Processing with Periodic Checkpointing ---
            # Using a generator keeps all workers 100% busy without artificial batch barriers,
            # while still allowing us to save progress periodically.
            save_frequency = 10 # Save a checkpoint every 10 completed tasks
            
            print(f"  Processing tasks continuously (saving checkpoint every {save_frequency} tasks)...")
            
            # "generator_unordered" yields results instantly, preventing a slow task from blocking the saves
            results_generator = Parallel(n_jobs=n_jobs, verbose=5, return_as="generator_unordered")(
                delayed(process_temperature_series_task)(task) for task in tasks_to_run
            )
            
            tasks_completed = 0
            pending_results = []
            
            for task_result in results_generator:
                pending_results.extend(task_result)
                tasks_completed += 1
                
                # Save to checkpoint when we hit the frequency threshold
                if tasks_completed % save_frequency == 0:
                    if pending_results:
                        df_batch = pd.DataFrame(pending_results)
                        if os.path.exists(checkpoint_filepath):
                            df_batch.to_csv(checkpoint_filepath, mode='a', header=False, index=False)
                        else:
                            df_batch.to_csv(checkpoint_filepath, index=False)
                        pending_results = [] # Reset for next save
                    print(f"    [Checkpoint] Safely wrote progress to disk (Task {tasks_completed})...")
                        
            # Save any remaining results after the loop finishes
            if pending_results:
                df_batch = pd.DataFrame(pending_results)
                if os.path.exists(checkpoint_filepath):
                    df_batch.to_csv(checkpoint_filepath, mode='a', header=False, index=False)
                else:
                    df_batch.to_csv(checkpoint_filepath, index=False)
            
            end_time = time.time()
            duration = end_time - start_time
            duration_formatted = str(datetime.timedelta(seconds=duration))
            print(f"Chunk calculations completed in {duration_formatted}.")
            
            # Load the complete data (from checkpoints)
            if not os.path.exists(checkpoint_filepath):
                print(f"No results generated for n_total={n_total}.")
                continue

            df = pd.read_csv(checkpoint_filepath)
            
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
                    df_patch = pd.DataFrame(flat_patch)
                    # Append patch results to checkpoint as well so we don't lose them
                    df_patch.to_csv(checkpoint_filepath, mode='a', header=False, index=False)
                    df = pd.concat([df, df_patch], ignore_index=True)
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
            
            # Remove checkpoint file after successful completion
            try:
                os.remove(checkpoint_filepath)
            except Exception as e:
                pass

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
            PhaseDiagramPlotting3Phase(master_filepath, save_dir=output_dir, timestamp=timestamp, auto_show=auto_show, apply_moving_average=False)
        except Exception as e:
            print(f"Warning: Could not open plots. Error: {e}")
        
if __name__ == "__main__":
    # --- USER-DEFINED SECTION ---
    
    # Select the run mode: "FULL_SIM", "PATCH", or "AUTOFIX"
    RUN_MODE = "FULL_SIM"

    # --- FULL SIMULATION CONFIGURATION ---
    # Used if RUN_MODE is "FULL_SIM".
    # config = standard_configuration
    full_run_config = standard_configuration

    # --- PATCHING CONFIGURATION ---
    # Used if RUN_MODE is "PATCH" or "AUTOFIX".
    
    # 1. Specify the file to patch.
    # Example: FILE_TO_PATCH = "Results/3Phase_LowRes_20231027_103000.csv"
    FILE_TO_MODIFY = "C:\\Users\\megu\\Documents\\VSCode 3 Phase 3D BNP\\Results worth saving\\n5e-17\\n5e-17.csv" # <-- CHANGE THIS

    # --- PATCHING-SPECIFIC CONFIGURATION ---
    # Used only if RUN_MODE is "PATCH".
    AREAS_TO_PATCH = [
        # {
        #     # In this area, only calculate Core-Shell, Liquid-Liquid, with no skin.
        #     'T_range': (750.0, 850.0),
        #     'xB_range': (0.25, 0.35),
        #     'filters': {
        #         'geometries': ['Core Shell'],
        #         'phase_pairs': [('Liquid', 'Liquid')],
        #         'skin_options': [False]
        #     }
        # },
        # {
        #     # In this area, only calculate Single Phase Liquid.
        #     'T_range': (1200.0, 1400.0),
        #     'xB_range': (0.0, 1.0),
        #     'filters': {
        #         'geometries': ['SinglePhase'],
        #         'single_phases': ['Liquid']
        #     }
        # },
        # {   # This area will run a full exhaustive search because no 'filters' key is provided.
        #     'T_range': (900.0, 1000.0),
        #     'xB_range': (0.70, 0.80)
        # }
        {
            'T_range': (500.0, 1150.0),
            'xB_range': (0.0, 0.75),
            'filters': {
                'geometries': ['Core Shell'],
                'phase_pairs': [('FCC', 'FCC')],
                'skin_options': [False]
            }
        },
    ]

    # --- EXECUTION ---
    if RUN_MODE == "FULL_SIM":
        processor = BNPSeriesProcessor(full_run_config)
        processor.run()
    elif RUN_MODE == "PATCH":
        patch_config = ThreePhaseConfiguration(base_file_name="patch_run")
        processor = BNPSeriesProcessor(patch_config)
        
        if FILE_TO_MODIFY and "YYYYMMDD" not in FILE_TO_MODIFY:
             processor.run_patch_on_file(
                filepath=FILE_TO_MODIFY,
                areas_to_patch=AREAS_TO_PATCH
            )
        else:
            print("\n--- PATCH MODE ---")
            print("Please edit the script to set RUN_MODE = 'PATCH'")
            print("and provide a valid FILE_TO_MODIFY path.")
    elif RUN_MODE == "AUTOFIX":
        patch_config = ThreePhaseConfiguration(base_file_name="autofix_run")
        processor = BNPSeriesProcessor(patch_config)
        
        if FILE_TO_MODIFY and "YYYYMMDD" not in FILE_TO_MODIFY:
             processor.run_speckle_autofix_on_file(
                filepath=FILE_TO_MODIFY
            )
        else:
            print("\n--- AUTOFIX MODE ---")
            print("Please edit the script to set RUN_MODE = 'AUTOFIX'")
            print("and provide a valid FILE_TO_MODIFY path.")