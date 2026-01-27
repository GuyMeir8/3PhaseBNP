import multiprocessing
import itertools
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from dataclasses import asdict
from generic_configuration_gibbs_free_energy import low_res_configuration, GeometryType
from Gibbs_free_energy_optimizer import GibbsOptimizer
from plotting_file import PhaseDiagramPlotting


def process_temperature_series(task_data):
    """
    Worker function. Calculates a series of Temperatures for a CONSTANT configuration.
    """
    xB, n_total, geometry, phase_pair, temperatures, config = task_data

    optimizer = GibbsOptimizer(config)
    results = []
    current_guess = [0.5, 0.5]

    for T in temperatures:
        res = optimizer.solve_specific_configuration(
            T=T, xB_total=xB, n_total=n_total,
            geometry=geometry, phase_pair=phase_pair,
            initial_guess=current_guess
        )

        # --- Format Data for Output ---
        res_dict = asdict(res)

        # 1. Basic Identifiers
        res_dict['T'] = T
        res_dict['xB_total'] = xB
        res_dict['n_total'] = n_total
        res_dict['GeoType'] = geometry.geometry_type.name
        res_dict['PhaseAlpha'] = phase_pair[0]
        res_dict['PhaseBeta'] = phase_pair[1]

        # 2. Outer Shell Distinction (for Plotting)
        # "None", "Liquid_A", "FCC_B", etc.
        if not geometry.has_outer_shell:
            res_dict['OuterShellType'] = "None"
        else:
            mat_name = config.a_data.name if geometry.shell.material == 0 else config.b_data.name
            res_dict['OuterShellType'] = f"{geometry.shell.phase}_{mat_name}"

        # Remove complex object to keep CSV clean
        del res_dict['geometry']

        results.append(res_dict)

        # Update guess for next step if valid and not single phase
        if not res.is_single_phase and res.G_min != float('inf'):
            current_guess = [res.ratio_A_alpha, res.ratio_B_alpha]

    return results


def generate_tasks(config):
    tasks = []
    phase_permutations = list(itertools.permutations(config.phase_options, 2))
    # Allow identical phases (e.g. FCC-FCC) for Core-Shell checks, if desired:
    phase_permutations += [(p, p) for p in config.phase_options]

    for xB in config.xb_values:
        for n_total in config.n_total_values:
            for geometry in config.geometry_options:

                # --- SKIP LOGIC FOR N=1 (Macroscopic) ---
                if n_total == 1.0:
                    # Skip anything with outer shell
                    if geometry.has_outer_shell:
                        continue
                    # Skip Core Shell (only do Janus)
                    if geometry.geometry_type == GeometryType.CORE_SHELL:
                        continue

                # --- SINGLE PHASE ---
                if geometry.geometry_type == GeometryType.SINGLE_PHASE:
                    for phase in config.phase_options:
                        tasks.append((xB, n_total, geometry, (phase, phase), config.temperature_values, config))

                # --- TWO PHASE ---
                else:
                    for pair in phase_permutations:
                        # Skip Liquid-Liquid
                        if pair[0] == 'Liquid' and pair[1] == 'Liquid':
                            continue

                        # Skip symmetric Janus without shell (A-B is same as B-A)
                        if geometry.geometry_type == GeometryType.JANUS or geometry.geometry_type == GeometryType.SPHERIC_JANUS:
                            if pair[0] > pair[1]:
                                continue

                        tasks.append((xB, n_total, geometry, pair, config.temperature_values, config))
    return tasks


def run_parallel_phase_diagram(config):
    tasks = generate_tasks(config)
    print(f"Generated {len(tasks)} parallel tasks.")

    # n_jobs=-1 uses all available cores
    list_of_lists = Parallel(n_jobs=-1, batch_size='auto', verbose=10)(
        delayed(process_temperature_series)(task) for task in tasks
    )

    all_data = []
    for sublist in list_of_lists:
        all_data.extend(sublist)

    df = pd.DataFrame(all_data)
    df = df[df['G_min'] != float('inf')]

    output_folder = "Results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    filename = config.get_output_file_name()
    full_path = os.path.join(output_folder, filename)
    df.to_csv(full_path, index=False)
    print(f"Saved {len(df)} rows to {filename}")
    print("Calculations complete. Generating phase diagrams...")
    PhaseDiagramPlotting(full_path)

if __name__ == "__main__":
    # joblib handles backend protection, usually don't need freeze_support explicitly
    # but good to keep if running as script
    run_parallel_phase_diagram(low_res_configuration)