import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from collections import Counter
from sklearn.linear_model import LinearRegression

# --- USER SETTINGS ---
PHASE_THRESHOLD = 0.005 # Phases making up less than 0.5% of the total are removed

def apply_threshold(df_in, threshold):
    df = df_in.copy()
    
    for c in ["n_total", "n_alpha", "n_beta"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    mask_not_single = df["Geometry"] != "SinglePhase"
    n_tot = df["n_total"]
    
    f_alpha = df["n_alpha"].fillna(0) / n_tot
    f_beta = df["n_beta"].fillna(0) / n_tot

    mask_cs_nano = (df["Geometry"] == "Core_Shell") & (abs(df["n_total"] - 1.0) > 1e-9)

    mask_tiny_alpha = mask_not_single & (f_alpha < threshold)
    df.loc[mask_tiny_alpha, "Geometry"] = "SinglePhase"
    df.loc[mask_tiny_alpha, "PhaseAlpha"] = df.loc[mask_tiny_alpha, "PhaseBeta"]
    df.loc[mask_tiny_alpha, "PhaseBeta"] = "None"
    df.loc[mask_tiny_alpha, "HasSkin"] = False
    df.loc[mask_tiny_alpha, "xB_skin"] = np.nan
    df.loc[mask_tiny_alpha, "n_alpha"] = df.loc[mask_tiny_alpha, "n_total"]
    df.loc[mask_tiny_alpha, "n_beta"] = np.nan

    mask_tiny_beta = mask_not_single & (f_beta < threshold) & (~mask_tiny_alpha) & (~mask_cs_nano)
    df.loc[mask_tiny_beta, "Geometry"] = "SinglePhase"
    df.loc[mask_tiny_beta, "PhaseBeta"] = "None"
    df.loc[mask_tiny_beta, "HasSkin"] = False
    df.loc[mask_tiny_beta, "xB_skin"] = np.nan
    df.loc[mask_tiny_beta, "n_alpha"] = df.loc[mask_tiny_beta, "n_total"]
    df.loc[mask_tiny_beta, "n_beta"] = np.nan

    return df

def identify_and_fix_suspect_points(df_in, max_iterations=10):
    """Flags categorical suspect points and overwrites them with the majority neighbor."""
    df = df_in.copy()
    df['Is_Suspect'] = False
    
    for n in df["n_total"].unique():
        mask_n = df["n_total"] == n
        subset = df[mask_n].copy() 
        
        subset['T_rounded'] = subset['T'].round(3)
        subset['xB_rounded'] = subset['xB_total'].round(4)
        
        T_vals = np.sort(subset['T_rounded'].unique())
        xB_vals = np.sort(subset['xB_rounded'].unique())

        if len(T_vals) < 3 or len(xB_vals) < 3:
            continue 

        T_to_idx = {t: i for i, t in enumerate(T_vals)}
        xB_to_idx = {x: i for i, x in enumerate(xB_vals)}

        for pass_num in range(1, max_iterations + 1):
            
            grid = np.empty((len(T_vals), len(xB_vals)), dtype=object)
            grid.fill(None)

            current_rows = df[mask_n]
            for idx, row in current_rows.iterrows():
                r = T_to_idx[subset.loc[idx, 'T_rounded']]
                c = xB_to_idx[subset.loc[idx, 'xB_rounded']]
                comp_str = f"{row['Geometry']}_{row['PhaseAlpha']}_{row['PhaseBeta']}_{row['HasSkin']}"
                grid[r, c] = (comp_str, idx, row['G_min'])

            rows, cols = grid.shape
            changes_to_apply = {} 

            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] is None: continue
                    val_str, target_idx, val_gmin = grid[r, c]
                    
                    is_top_or_bottom = (r == 0) or (r == rows - 1)
                    is_left_or_right = (c == 0) or (c == cols - 1)
                    is_corner = is_top_or_bottom and is_left_or_right
                    is_edge = (is_top_or_bottom or is_left_or_right) and not is_corner
                    
                    neighbor_list = []
                    
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if grid[nr, nc] is not None:
                                    neighbor_list.append(grid[nr, nc])
                                    
                    if not neighbor_list: continue
                    
                    counts = Counter([n[0] for n in neighbor_list])
                    val_str_count = counts.get(val_str, 0)
                    
                    other_counts = {k: v for k, v in counts.items() if k != val_str}
                    if not other_counts: continue 
                    
                    max_other_count = max(other_counts.values())
                    tied_candidates = [k for k, v in other_counts.items() if v == max_other_count]
                    
                    should_change = False
                    if is_corner:
                        if max_other_count >= 3: should_change = True
                    elif is_edge:
                        if max_other_count >= 4: should_change = True
                    else: 
                        if max_other_count >= 5: 
                            should_change = True
                        elif val_str_count <= 1: 
                            should_change = True
                    
                    if should_change:
                        if len(tied_candidates) == 1:
                            best_cand = tied_candidates[0]
                        else:
                            # TIE-BREAKER: Lowest average G_min
                            avg_gmin = {
                                cand: np.mean([n[2] for n in neighbor_list if n[0] == cand])
                                for cand in tied_candidates
                            }
                            best_cand = min(avg_gmin, key=avg_gmin.get)
                            
                        matching_indices = [n[1] for n in neighbor_list if n[0] == best_cand]
                        
                        changes_to_apply[target_idx] = {
                            'cat': matching_indices[0], 
                            'nums': matching_indices
                        }

            if not changes_to_apply:
                if pass_num > 1:
                    print(f"  n={n:.1e} (Cat): Reached steady state after {pass_num - 1} passes.")
                else:
                    print(f"  n={n:.1e} (Cat): No suspect points found on first pass.")
                break

            print(f"  n={n:.1e} (Cat): Pass {pass_num} fixing {len(changes_to_apply)} points...")

            cat_cols = ['Geometry', 'PhaseAlpha', 'PhaseBeta', 'HasSkin']
            num_cols = ['xB_skin', 'n_alpha', 'n_beta', 'xB_alpha', 'xB_beta']
            
            for target_idx, info in changes_to_apply.items():
                df.loc[target_idx, cat_cols] = df.loc[info['cat'], cat_cols]
                
                # Only average columns that actually exist in the dataframe to avoid errors
                valid_num_cols = [c for c in num_cols if c in df.columns]
                df.loc[target_idx, valid_num_cols] = df.loc[info['nums'], valid_num_cols].mean()
                df.loc[target_idx, 'Is_Suspect'] = True

    return df

def get_r2(x, y):
    """Calculates the R^2 value for a set of points to measure linearity."""
    if len(x) < 2: 
        return 1.0
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    
    # Handle perfectly flat regions without raising errors
    if np.all(y == y[0]):
        return 1.0
        
    try:
        model = LinearRegression().fit(x, y)
        return model.score(x, y)
    except:
        return 1.0

def is_r2_anomaly(vals):
    """
    Checks if the center point (index 2) significantly degrades the R^2 
    of the 5-point set compared to the line formed by its neighbors.
    """
    if any(pd.isna(v) for v in vals): 
        return False
    
    # R2 of the 4 neighbors only (excluding the target center point)
    x_neighbors = np.array([0, 1, 3, 4])
    y_neighbors = np.array([vals[0], vals[1], vals[3], vals[4]])
    r2_neighbors = get_r2(x_neighbors, y_neighbors)
    
    # R2 of all 5 points (including the target center point)
    x_full = np.array([0, 1, 2, 3, 4])
    y_full = np.array(vals)
    r2_full = get_r2(x_full, y_full)
    
    # TRIGGER: If neighbors are highly linear (>0.9) but the center point 
    # causes the R2 to drop by more than 10% (0.1), it is a suspect spike.
    if r2_neighbors > 0.9 and (r2_neighbors - r2_full) > 0.1:
        return True
        
    return False

def smooth_slope_anomalies(df_in, max_iterations=10):
    """Detects and fixes numerical noise within continuous phase regions using R^2 analysis."""
    df = df_in.copy()
    
    for n in df["n_total"].unique():
        mask_n = df["n_total"] == n
        subset = df[mask_n].copy() 
        
        subset['T_rounded'] = subset['T'].round(3)
        subset['xB_rounded'] = subset['xB_total'].round(4)
        
        T_vals = np.sort(subset['T_rounded'].unique())
        xB_vals = np.sort(subset['xB_rounded'].unique())

        if len(T_vals) < 5 or len(xB_vals) < 5:
            continue 

        T_to_idx = {t: i for i, t in enumerate(T_vals)}
        xB_to_idx = {x: i for i, x in enumerate(xB_vals)}

        for pass_num in range(1, max_iterations + 1):
            
            grid = np.empty((len(T_vals), len(xB_vals)), dtype=object)
            grid.fill(None)

            current_rows = df[mask_n]
            for idx, row in current_rows.iterrows():
                r = T_to_idx[subset.loc[idx, 'T_rounded']]
                c = xB_to_idx[subset.loc[idx, 'xB_rounded']]
                comp_str = f"{row['Geometry']}_{row['PhaseAlpha']}_{row['PhaseBeta']}_{row['HasSkin']}"
                
                # Provide fractional values for R2 analysis
                f_alpha = row['n_alpha'] / n if pd.notna(row['n_alpha']) else np.nan
                f_beta = row['n_beta'] / n if pd.notna(row['n_beta']) else np.nan
                
                # Use .get() to safely handle cases where phase compositions might be missing
                xb_a = row.get('xB_alpha', np.nan)
                xb_b = row.get('xB_beta', np.nan)
                xb_s = row.get('xB_skin', np.nan)
                
                grid[r, c] = (comp_str, idx, f_alpha, f_beta, xb_a, xb_b, xb_s)

            rows, cols = grid.shape
            changes_to_apply = {} 

            # 1. Horizontal sweep
            for r in range(rows):
                for c in range(2, cols - 2):
                    pts = [grid[r, c+i] for i in range(-2, 3)]
                    if None in pts: continue
                    
                    if len(set(p[0] for p in pts)) != 1: continue 
                    
                    f_alphas = [p[2] for p in pts]
                    f_betas = [p[3] for p in pts]
                    xb_alphas = [p[4] for p in pts]
                    xb_betas = [p[5] for p in pts]
                    xb_skins = [p[6] for p in pts]
                    
                    target_idx = pts[2][1]
                    neighbor_indices = [pts[i][1] for i in [0, 1, 3, 4]]
                    
                    if (is_r2_anomaly(f_alphas) or is_r2_anomaly(f_betas) or 
                        is_r2_anomaly(xb_alphas) or is_r2_anomaly(xb_betas) or 
                        is_r2_anomaly(xb_skins)):
                        changes_to_apply[target_idx] = neighbor_indices

            # 2. Vertical sweep
            for c in range(cols):
                for r in range(2, rows - 2):
                    pts = [grid[r+i, c] for i in range(-2, 3)]
                    if None in pts: continue
                    
                    if len(set(p[0] for p in pts)) != 1: continue
                    
                    f_alphas = [p[2] for p in pts]
                    f_betas = [p[3] for p in pts]
                    xb_alphas = [p[4] for p in pts]
                    xb_betas = [p[5] for p in pts]
                    xb_skins = [p[6] for p in pts]
                    
                    target_idx = pts[2][1]
                    neighbor_indices = [pts[i][1] for i in [0, 1, 3, 4]]
                    
                    if target_idx not in changes_to_apply:
                        if (is_r2_anomaly(f_alphas) or is_r2_anomaly(f_betas) or 
                            is_r2_anomaly(xb_alphas) or is_r2_anomaly(xb_betas) or 
                            is_r2_anomaly(xb_skins)):
                            changes_to_apply[target_idx] = neighbor_indices

            if not changes_to_apply:
                if pass_num > 1:
                    print(f"  n={n:.1e} (R2-Slope): Reached steady state after {pass_num - 1} passes.")
                else:
                    print(f"  n={n:.1e} (R2-Slope): No numerical anomalies found on first pass.")
                break

            print(f"  n={n:.1e} (R2-Slope): Pass {pass_num} smoothing {len(changes_to_apply)} points...")

            num_cols = ['xB_skin', 'n_alpha', 'n_beta', 'xB_alpha', 'xB_beta']
            
            for target_idx, neighbor_indices in changes_to_apply.items():
                valid_num_cols = [c for c in num_cols if c in df.columns]
                df.loc[target_idx, valid_num_cols] = df.loc[neighbor_indices, valid_num_cols].mean()
                df.loc[target_idx, 'Is_Suspect'] = True

    return df

def main():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw() 

    file_path = filedialog.askopenfilename(
        title="Select raw data CSV file to clean",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Selected file: {file_path}")

    try:
        df_raw = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("Filtering valid results and finding minimum energy configurations...")
    df_valid = df_raw[(df_raw["G_min"] != 1.0) & (~np.isinf(df_raw["G_min"]))].copy()
    
    if df_valid.empty:
        print("No valid results found in file.")
        return

    idx = df_valid.groupby(["n_total", "xB_total", "T"])["G_min"].idxmin()
    df_min = df_valid.loc[idx].copy()

    print(f"Applying phase threshold: {PHASE_THRESHOLD}")
    df_cleaned = apply_threshold(df_min, PHASE_THRESHOLD)
    
    print("\n--- Running Categorical Speckle Remover ---")
    df_cat_fixed = identify_and_fix_suspect_points(df_cleaned, max_iterations=10)
    
    print("\n--- Running Numerical Slope Smoother ---")
    df_fully_fixed = smooth_slope_anomalies(df_cat_fixed, max_iterations=10)
    
    num_suspects = df_fully_fixed["Is_Suspect"].sum() if "Is_Suspect" in df_fully_fixed else 0
    print(f"\nTotal points modified across all passes: {num_suspects}")

    base_name, ext = os.path.splitext(file_path)
    new_file_path = f"{base_name}_cleaned{ext}"

    df_fully_fixed.to_csv(new_file_path, index=False)
    print(f"File saved as: {new_file_path}")

if __name__ == "__main__":
    main()