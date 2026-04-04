import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from collections import Counter

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
                
                valid_num_cols = [c for c in num_cols if c in df.columns]
                df.loc[target_idx, valid_num_cols] = df.loc[info['nums'], valid_num_cols].mean()
                df.loc[target_idx, 'Is_Suspect'] = True

    return df

# --- NEW 15% AVERAGE LOGIC ---

def is_spike_anomaly(vals, mode):
    """
    Calculates the average of the neighbors (ignoring the target point).
    If the target point deviates by more than 15% of that average, flags it as a spike.
    """
    if any(pd.isna(v) for v in vals): return False
    
    if mode == 'sym':
        # Neighbors: [0, 1, 3, 4], Target: [2]
        navg = (vals[0] + vals[1] + vals[3] + vals[4]) / 4.0
        target = vals[2]
    elif mode == 'edge_0':
        # Neighbors: [1, 2, 3], Target: [0]
        navg = (vals[1] + vals[2] + vals[3]) / 3.0
        target = vals[0]
    elif mode == 'edge_3':
        # Neighbors: [0, 1, 2], Target: [3]
        navg = (vals[0] + vals[1] + vals[2]) / 3.0
        target = vals[3]
        
    # Margin is 15% of the local average (with a 0.01 absolute floor for zero-noise)
    margin = max(0.15 * abs(navg), 0.01)
    
    if abs(target - navg) > margin:
        return True
    return False

def check_pts_for_anomaly(pts, mode):
    """Helper to sweep all numerical attributes for a 15% spike anomaly."""
    for attr_idx in [2, 3, 4, 5, 6]:
        vals = [p[attr_idx] for p in pts]
        if is_spike_anomaly(vals, mode): 
            return True
    return False

def is_valid_window(pts):
    """Helper to ensure all points in a window exist and share the same phase typing."""
    if any(p is None for p in pts): return False
    return len(set(p[0] for p in pts)) == 1

def is_applicable_border(window_pts, outside_pt):
    """
    Checks if a boundary involves a SinglePhase region OR any Liquid phase.
    Allows the edge case where the boundary is the physical edge of the grid.
    """
    win_str = window_pts[0][0]
    if "SinglePhase" in win_str or "Liquid" in win_str:
        return True
        
    if outside_pt is None:
        return True 
        
    out_str = outside_pt[0]
    if "SinglePhase" in out_str or "Liquid" in out_str:
        return True
        
    return False

def smooth_numerical_anomalies(df_in, max_iterations=10):
    """Detects and fixes numerical spikes using the 15% neighbor average rule."""
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
                
                f_alpha = row['n_alpha'] / n if pd.notna(row['n_alpha']) else np.nan
                f_beta = row['n_beta'] / n if pd.notna(row['n_beta']) else np.nan
                xb_a = row.get('xB_alpha', np.nan)
                xb_b = row.get('xB_beta', np.nan)
                xb_s = row.get('xB_skin', np.nan)
                
                grid[r, c] = (comp_str, idx, f_alpha, f_beta, xb_a, xb_b, xb_s)

            rows, cols = grid.shape
            changes_to_apply = {} 

            # 1. Horizontal sweep
            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] is None: continue
                    target_idx = grid[r, c][1]
                    if target_idx in changes_to_apply: continue

                    sym_pts = [grid[r, c+i] if 0 <= c+i < cols else None for i in range(-2, 3)]
                    
                    if is_valid_window(sym_pts):
                        if check_pts_for_anomaly(sym_pts, 'sym'):
                            # Overwrite with the 4 neighbors
                            changes_to_apply[target_idx] = [sym_pts[i][1] for i in [0, 1, 3, 4]]
                    else:
                        right_pts = [grid[r, c+i] if 0 <= c+i < cols else None for i in range(4)]
                        left_pts = [grid[r, c+i] if 0 <= c+i < cols else None for i in range(-3, 1)]
                        
                        outside_left = grid[r, c-1] if 0 <= c-1 < cols else None
                        outside_right = grid[r, c+1] if 0 <= c+1 < cols else None
                        
                        if is_valid_window(right_pts) and is_applicable_border(right_pts, outside_left):
                            if check_pts_for_anomaly(right_pts, 'edge_0'):
                                changes_to_apply[target_idx] = [right_pts[i][1] for i in [1, 2, 3]]
                        elif is_valid_window(left_pts) and is_applicable_border(left_pts, outside_right):
                            if check_pts_for_anomaly(left_pts, 'edge_3'):
                                changes_to_apply[target_idx] = [left_pts[i][1] for i in [0, 1, 2]]

            # 2. Vertical sweep
            for c in range(cols):
                for r in range(rows):
                    if grid[r, c] is None: continue
                    target_idx = grid[r, c][1]
                    if target_idx in changes_to_apply: continue

                    sym_pts = [grid[r+i, c] if 0 <= r+i < rows else None for i in range(-2, 3)]
                    
                    if is_valid_window(sym_pts):
                        if check_pts_for_anomaly(sym_pts, 'sym'):
                            changes_to_apply[target_idx] = [sym_pts[i][1] for i in [0, 1, 3, 4]]
                    else:
                        down_pts = [grid[r+i, c] if 0 <= r+i < rows else None for i in range(4)]
                        up_pts = [grid[r+i, c] if 0 <= r+i < rows else None for i in range(-3, 1)]
                        
                        outside_up = grid[r-1, c] if 0 <= r-1 < rows else None
                        outside_down = grid[r+1, c] if 0 <= r+1 < rows else None
                        
                        if is_valid_window(down_pts) and is_applicable_border(down_pts, outside_up):
                            if check_pts_for_anomaly(down_pts, 'edge_0'):
                                changes_to_apply[target_idx] = [down_pts[i][1] for i in [1, 2, 3]]
                        elif is_valid_window(up_pts) and is_applicable_border(up_pts, outside_down):
                            if check_pts_for_anomaly(up_pts, 'edge_3'):
                                changes_to_apply[target_idx] = [up_pts[i][1] for i in [0, 1, 2]]

            if not changes_to_apply:
                if pass_num > 1:
                    print(f"  n={n:.1e} (Avg-Slope): Reached steady state after {pass_num - 1} passes.")
                else:
                    print(f"  n={n:.1e} (Avg-Slope): No numerical anomalies found on first pass.")
                break

            print(f"  n={n:.1e} (Avg-Slope): Pass {pass_num} smoothing {len(changes_to_apply)} points...")

            num_cols = ['xB_skin', 'n_alpha', 'n_beta', 'xB_alpha', 'xB_beta']
            
            for target_idx, neighbor_indices in changes_to_apply.items():
                valid_num_cols = [c for c in num_cols if c in df.columns]
                # Because we pass the neighbor indices to pandas, it natively calculates the 
                # physical average of the raw moles/compositions without scaling issues!
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
    
    print("\n--- Running Numerical Spike Smoother ---")
    df_fully_fixed = smooth_numerical_anomalies(df_cat_fixed, max_iterations=10)
    
    num_suspects = df_fully_fixed["Is_Suspect"].sum() if "Is_Suspect" in df_fully_fixed else 0
    print(f"\nTotal points modified across all passes: {num_suspects}")

    base_name, ext = os.path.splitext(file_path)
    new_file_path = f"{base_name}_cleaned{ext}"

    df_fully_fixed.to_csv(new_file_path, index=False)
    print(f"File saved as: {new_file_path}")

if __name__ == "__main__":
    main()