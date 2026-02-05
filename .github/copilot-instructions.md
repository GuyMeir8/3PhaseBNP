## Purpose
Short, actionable instructions for AI coding agents working in this repo.

## Quick run
- Install: `pip install -r requirements.txt` (or install `numpy scipy pandas matplotlib seaborn`)
- Run: `python main.py` — picks a low-res test config and will attempt to plot latest CSV from `Results/`.

## Big-picture architecture
- Entry: [main.py](main.py) wires a `ThreePhaseConfiguration` -> `SystemData` -> `BNPOptimizer3Phase` -> `GibbsEnergyCalculator3Phase` and plotting via [plotting_3_phase.py](plotting_3_phase.py).
- Data flow: `configurations_3_phase.py` defines simulation grids and overrides; `SystemData` dynamically loads material and interaction classes from `pure_material_properties.py` and `interaction_properties.py` and populates `SurfaceEnergyValues`.
- Core logic: [BNP_Gibbs_en_calc_3_phase.py](BNP_Gibbs_en_calc_3_phase.py) does energy calculations (ideal/excess/surface) and geometry solvers (`calc_Janus...`, `calc_core_shell...`).
- Optimization: [BNP_optimizer_3_phase.py](BNP_optimizer_3_phase.py) runs stratified scouting + constrained `SLSQP` minimizations and returns `OptimizationResult3Phase`.

## Project-specific conventions and patterns
- Dynamic discovery: `SystemData._load_data_from_module` instantiates classes that subclass `BaseMaterial` / `BaseInteraction` and uses their `.name` attribute as the key. Do not rename or change `.name` without updating callers.
- Surface energy overrides: `ThreePhaseConfiguration.surface_energy_overrides` keys are tuples like `("Ag","FCC","Cu","Liquid")` and values are either a float or a function `T -> energy`. See [`configurations_3_phase.py`](configurations_3_phase.py) for an example.
- Failure signalling: the energy calculator returns `1.0` on failure; plotting filters `G_min == 1.0` as invalid results. Keep this sentinel in mind when changing error paths.
- Geometry strings: valid `geometry_type` values are `"Janus"` and `"Core_Shell"` (and `SinglePhase` as a plotting label). Phase names observed: `"FCC"`, `"Liquid"`.
- Numerical guards: optimizer avoids exact 0/1 by using `eps=1e-4`; guesses include near-1 values (0.995, 0.999) to explore solubility edges — do not remove without validation.

## Important integration points
- `pure_material_properties.py` and `interaction_properties.py`: add new materials/interactions by subclassing the provided base classes; ensure `phases` dict keys match the phase names used elsewhere.
- `SurfaceEnergyValues` structure: `system_data.surface_energy.AaBb[matA][phaseA][matB][phaseB]` stores callables; overrides replace these entries with callables.
- Output/plotting: results are CSV rows (see examples in `Results/`). Plotting expects columns like `n_total`, `xB_total`, `T`, `G_min`, `Geometry`, `PhaseAlpha`, `PhaseBeta`, `HasSkin`, `xB_skin`, `n_alpha`, `n_beta`.

## When editing numeric code
- Prefer small, localized changes. Add unit-like checks for geometry solvers: many branches raise `ValueError` on unphysical geometry — tests should assert those cases.
- If changing convergence tolerances, run a quick grid (low_res configuration in [configurations_3_phase.py](configurations_3_phase.py)) to confirm no large-scale behavior changes.

## Local developer workflows
- Generate quick results: use `debug_configuration` in [configurations_3_phase.py](configurations_3_phase.py) to run a single-point fast check.
- Reproduce plotting: drop a CSV into `Results/` and run `python plotting_3_phase.py` (or `python main.py` which will plot the most recent file).

## Where to look for examples
- Surface-energy override example: [configurations_3_phase.py](configurations_3_phase.py)
- Optimization strategy + candidate map: [BNP_optimizer_3_phase.py](BNP_optimizer_3_phase.py) (search for "THE SCOUT" / "THE SNIPER").
- Geometry solvers and error handling: [BNP_Gibbs_en_calc_3_phase.py](BNP_Gibbs_en_calc_3_phase.py) (`calc_Janus_geometry_for_known_nx`, `calc_core_shell_geometry_for_known_nx`).

If anything here is unclear or you'd like more detail (examples, line references, or additional conventions), tell me which area to expand.  