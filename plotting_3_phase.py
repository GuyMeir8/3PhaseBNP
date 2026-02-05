import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import glob
import os
import datetime

class PhaseDiagramPlotting3Phase:
    def __init__(self, file_name, save_dir=None, timestamp=None):
        # save_dir: optional directory to save figures to. If None, figures are only displayed.
        # timestamp: optional timestamp string to include in saved filenames
        
        if save_dir is None:
            self.save_dir = "Results"
        else:
            self.save_dir = save_dir

        if timestamp is None:
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.timestamp = timestamp
        
        # 1. Define Colors per User Request
        self.COLORS = {
            # Single Phase
            "Single_Liquid": "lightblue",
            "Single_FCC": "#FFDB58", # Mustard

            # Two Phase (Alpha/Beta)
            "Alpha_FCC": "lightgreen",
            "Beta_FCC": "darkgreen",
            "Alpha_Liquid": "#B57EDC", # Lavender (darkened slightly for visibility)
            "Beta_Liquid": "darkviolet",

            # Skin Outlines
            "Skin_A": "red",
            "Skin_B": "blue"
        }

        # 2. Load Data
        self.df = pd.read_csv(file_name)
        self.preprocess_data()

        # 3. Filter Valid Results
        # Remove failed calculations (G_min = 1.0 is the error flag)
        self.df = self.df[self.df["G_min"] != 1.0].copy()

        # 4. Find Minimum Energy Configuration per Point
        # Group by n_total, xB_total, T and find the row with the absolute minimum G_min
        idx = self.df.groupby(["n_total", "xB_total", "T"])["G_min"].idxmin()
        self.df_min = self.df.loc[idx].copy()

        # 5. Clean Dust / Enforce Thresholds
        # Converts Core-Shell with tiny cores/shells into SinglePhase for cleaner plotting
        self.df_min = self.df_min.apply(self.enforce_phase_threshold, axis=1)

        # 6. Generate Labels
        self.df_min["phase_label"] = self.df_min.apply(self.make_label, axis=1)

        # 7. Remove Speckles (Isolated single-phase points)
        cleaned_dfs = []
        for n in self.df_min["n_total"].unique():
            subset = self.df_min[self.df_min["n_total"] == n].copy()
            cleaned_subset = self.remove_speckles_from_slice(subset, passes=100)
            cleaned_dfs.append(cleaned_subset)
        self.df_min = pd.concat(cleaned_dfs, ignore_index=True)

        # 8. Plot
        self.plot_all_phase_diagrams_per_n_total()

    def preprocess_data(self):
        # Ensure string columns are stripped of whitespace
        for col in ["Geometry", "PhaseAlpha", "PhaseBeta"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
        
        # Ensure numeric columns
        cols = ["xB_total", "T", "G_min", "n_total", "n_alpha", "n_beta", "xB_skin"]
        for c in cols:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

    def make_label(self, row):
        geo = row["Geometry"]
        a = row["PhaseAlpha"]
        b = row["PhaseBeta"]
        has_skin = row["HasSkin"]
        xb_skin = row["xB_skin"]

        suffix = ""
        if has_skin and not pd.isna(xb_skin):
            if xb_skin < 0.5:
                suffix = "+SkinA"
            else:
                suffix = "+SkinB"
        
        if geo == "SinglePhase":
            return f"{geo}{suffix}__{a}"
        
        return f"{geo}{suffix}__{a}+{b}"

    def remove_speckles_from_slice(self, df_slice, passes=1):
        """
        Removes isolated speckles using a grid-based cellular automaton approach.
        Updates Geometry, PhaseAlpha, PhaseBeta, HasSkin based on the smoothed label.
        """
        try:
                        # 1. Pivot to Grid
            # Ensure duplicates are handled (though df_min should be unique per T, xB)
            try:
                grid_df = df_slice.pivot(index="T", columns="xB_total", values="phase_label")
            except ValueError:
                grid_df = df_slice.pivot_table(index="T", columns="xB_total", values="phase_label", aggfunc='first')

            grid = grid_df.values
            rows, cols = grid.shape


            # 2. Cellular Automaton Smoothing
            for i in range(passes):
                new_grid = grid.copy()
                changes_made = 0
                for r in range(rows):
                    for c in range(cols):
                        current_val = grid[r, c]
                        if pd.isna(current_val): continue
                        
                        valid_neighbors = []
                        neighbors_matching_self = 0
                        
                        # Check 8 neighbors
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = r + dr, c + dc
                                
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    val = grid[nr, nc]
                                    if pd.notna(val):
                                        valid_neighbors.append(val)
                                        if val == current_val:
                                            neighbors_matching_self += 1
                        
                        if not valid_neighbors: continue
                        
                        # Rule: If isolated (<= 1 matching neighbor), adopt majority neighbor
                        # Stricter for edges (<= 0)
                        is_edge = (r == 0) or (r == rows - 1) or (c == 0) or (c == cols - 1)
                        limit = 0 if is_edge else 1
                        
                        if neighbors_matching_self <= limit:
                            from collections import Counter
                            counts = Counter(valid_neighbors)
                            most_common_label, _ = counts.most_common(1)[0]
                            
                            if most_common_label != current_val:
                                new_grid[r, c] = most_common_label
                                changes_made += 1
                
                grid = new_grid
                if changes_made == 0:
                    break

            # 3. Map back to DataFrame
            cleaned_flat = pd.DataFrame(grid, index=grid_df.index, columns=grid_df.columns).reset_index().melt(id_vars="T", value_name="new_phase_label")
            
            # Merge with original slice
            df_merged = pd.merge(df_slice, cleaned_flat, on=["T", "xB_total"], how="left")
            
            # Identify rows that changed
            mask_changed = df_merged["phase_label"] != df_merged["new_phase_label"]
            change_count = mask_changed.sum()
            
            if change_count > 0:
                df_merged.loc[mask_changed, "phase_label"] = df_merged.loc[mask_changed, "new_phase_label"]
                
                # Update dependent columns for changed rows
                for idx in df_merged[mask_changed].index:
                    new_label = df_merged.loc[idx, "phase_label"]
                    geo, pa, pb, skin, xb_s = self._parse_label_to_cols(new_label)
                    
                    df_merged.loc[idx, "Geometry"] = geo
                    df_merged.loc[idx, "PhaseAlpha"] = pa
                    df_merged.loc[idx, "PhaseBeta"] = pb
                    df_merged.loc[idx, "HasSkin"] = skin
                    if skin:
                        df_merged.loc[idx, "xB_skin"] = xb_s
            
            df_merged.drop(columns=["new_phase_label"], inplace=True)
            return df_merged
            
        except Exception as e:
            print(f"  Error in neighbor-based speckle removal: {e}")
            import traceback
            traceback.print_exc()
            return df_slice

    def _parse_label_to_cols(self, label):
        # Helper to reverse make_label
        if pd.isna(label): return "SinglePhase", "Unknown", "None", False, np.nan
        label = str(label)
        
        # Split Geometry+Skin and Phases
        if "__" in label:
            left, right = label.split("__", 1)
        else:
            left, right = label, ""
            
        # Parse Skin
        has_skin = False
        xb_skin = np.nan
        
        if "+SkinA" in left:
            has_skin = True
            xb_skin = 0.0
            geo = left.replace("+SkinA", "")
        elif "+SkinB" in left:
            has_skin = True
            xb_skin = 1.0
            geo = left.replace("+SkinB", "")
        elif "+Skin" in left:
            has_skin = True
            xb_skin = 0.5
            geo = left.replace("+Skin", "")
        else:
            geo = left
            
        # Parse Phases
        if "+" in right:
            pa, pb = right.split("+", 1)
        else:
            pa = right
            pb = "None"
            
        return geo, pa, pb, has_skin, xb_skin

    def enforce_phase_threshold(self, row):
        # 1. Define Threshold
        THRESHOLD = 0.01 
        
        if row["Geometry"] == "SinglePhase":
            return row
        
        n_tot = row["n_total"]
        na = row["n_alpha"]
        nb = row["n_beta"]
        
        # --- FIX STARTS HERE ---
        # If calculation failed (NaN) or data is missing
        if pd.isna(na) or pd.isna(nb) or n_tot == 0:
            # Capture the original Beta phase type (e.g., "FCC" or "Liquid") BEFORE overwriting it
            original_beta_phase = row["PhaseBeta"]

            row["Geometry"] = "SinglePhase"
            row["PhaseBeta"] = "None" 

            if row["xB_total"] > 0.5:
                # Dominant B -> Single Phase Beta. Use the captured value.
                row["PhaseAlpha"] = original_beta_phase
            # Else Dominant A -> Single Phase Alpha (already set in PhaseAlpha)
            
            row["HasSkin"] = False
            return row
        # -----------------------

        # If Core (Alpha) is tiny -> SinglePhase Beta
        if (na / n_tot) < THRESHOLD:
            # Capture Beta phase type
            original_beta_phase = row["PhaseBeta"]
            
            row["Geometry"] = "SinglePhase"
            # Set Alpha to be the type of Beta (since Beta is the only thing left)
            row["PhaseAlpha"] = original_beta_phase
            row["PhaseBeta"] = "None"
            row["HasSkin"] = False
            return row
        
        # If Shell (Beta) is tiny -> SinglePhase Alpha
        if (nb / n_tot) < THRESHOLD:
            row["Geometry"] = "SinglePhase"
            row["PhaseBeta"] = "None"
            row["HasSkin"] = False
            return row
            
        return row

    def plot_all_phase_diagrams_per_n_total(self):
        df_plot = self.df_min
        
        for n in df_plot["n_total"].unique():
            subset = df_plot[df_plot["n_total"] == n]
            if subset.empty: continue
            
            plt.figure(figsize=(12, 8))
            base_size = 35
            MIN_VISIBLE_RATIO = 0.2 # Ensures a phase takes up at least 20% of the marker area

            # ---------------------------------------------------------
            # 1. Single Phase
            # ---------------------------------------------------------
            df_single = subset[subset["Geometry"] == "SinglePhase"].copy()
            if not df_single.empty:
                # Determine Colors
                colors = df_single["PhaseAlpha"].map({
                    "Liquid": self.COLORS["Single_Liquid"],
                    "FCC": self.COLORS["Single_FCC"]
                })
                
                # Determine Skin Outline
                edge_colors, line_widths = self._get_skin_styles(df_single)

                plt.scatter(
                    df_single["xB_total"], df_single["T"],
                    c=colors,
                    edgecolors=edge_colors,
                    linewidths=line_widths,
                    s=base_size,
                    marker='o',
                    label="Single Phase"
                )

            # ---------------------------------------------------------
            # 2. Core-Shell
            # ---------------------------------------------------------
            df_cs = subset[subset["Geometry"] == "Core_Shell"].copy()
            if not df_cs.empty:
                # Calculate Ratios
                df_cs["alpha_ratio"] = df_cs["n_alpha"] / df_cs["n_total"]
                df_cs["alpha_ratio"] = df_cs["alpha_ratio"].fillna(0)
                # Enforce minimum visible size for both core and shell for plotting
                plot_alpha_ratio = df_cs["alpha_ratio"].clip(lower=MIN_VISIBLE_RATIO, upper=1.0 - MIN_VISIBLE_RATIO)
                
                # Colors
                c_alpha = self._get_phase_colors(df_cs["PhaseAlpha"], "Alpha")
                c_beta = self._get_phase_colors(df_cs["PhaseBeta"], "Beta")
                
                # Skin Outline (Applied to Outer Shell / Beta)
                edge_colors, line_widths = self._get_skin_styles(df_cs)

                # Layer 1: Outer Shell (Beta) - Full Size
                plt.scatter(
                    df_cs["xB_total"], df_cs["T"],
                    c=c_beta,
                    edgecolors=edge_colors,
                    linewidths=line_widths,
                    s=base_size,
                    marker='o'
                )

                # Layer 2: Inner Core (Alpha) - Scaled Size
                # Size is area, so s_inner = s_total * ratio
                s_inner = base_size * plot_alpha_ratio
                plt.scatter(
                    df_cs["xB_total"], df_cs["T"],
                    c=c_alpha,
                    edgecolors='none',
                    s=s_inner,
                    marker='o',
                    label="Core-Shell"
                )

            # ---------------------------------------------------------
            # 3. Janus
            # ---------------------------------------------------------
            df_janus = subset[subset["Geometry"] == "Janus"].copy()
            if not df_janus.empty:
                df_janus["alpha_ratio"] = df_janus["n_alpha"] / df_janus["n_total"]
                
                # Enforce minimum visible size for both phases for plotting
                plot_alpha_ratio = df_janus["alpha_ratio"].clip(lower=MIN_VISIBLE_RATIO, upper=1.0 - MIN_VISIBLE_RATIO)
                
                c_alpha = self._get_phase_colors(df_janus["PhaseAlpha"], "Alpha")
                c_beta = self._get_phase_colors(df_janus["PhaseBeta"], "Beta")
                edge_colors, line_widths = self._get_skin_styles(df_janus)

                # Layer 1: Background (Beta) - Full Circle
                plt.scatter(
                    df_janus["xB_total"], df_janus["T"],
                    c=c_beta,
                    edgecolors=edge_colors,
                    linewidths=line_widths,
                    s=base_size,
                    marker='o'
                )

                # Layer 2: Foreground (Alpha) - Segment Marker
                # Bin ratios to avoid creating unique markers for every point
                df_janus["ratio_bin"] = (plot_alpha_ratio * 20).round() / 20 # Nearest 0.05
                
                for ratio in df_janus["ratio_bin"].unique():
                    mask = df_janus["ratio_bin"] == ratio
                    chunk = df_janus[mask]
                    
                    marker_verts = self._create_segment_marker(ratio)
                    
                    plt.scatter(
                        chunk["xB_total"], chunk["T"],
                        c=c_alpha[mask],
                        edgecolors='none',
                        s=base_size, # Marker is unit circle, scales to base_size
                        marker=marker_verts,
                        label="Janus" if ratio == df_janus["ratio_bin"].unique()[0] else ""
                    )

            plt.title(f"Phase Diagram (n={n:.1e})")
            plt.xlabel("xB_total (Composition)")
            plt.ylabel("Temperature [K]")
            
            # Custom Legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Single Phase (Liquid)',
                       markerfacecolor=self.COLORS['Single_Liquid'], markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Single Phase (FCC)',
                       markerfacecolor=self.COLORS['Single_FCC'], markersize=10),
                
                Patch(facecolor=self.COLORS['Alpha_FCC'], label='Alpha (Core/Seg): FCC'),
                Patch(facecolor=self.COLORS['Beta_FCC'], label='Beta (Shell/Back): FCC'),
                Patch(facecolor=self.COLORS['Alpha_Liquid'], label='Alpha (Core/Seg): Liquid'),
                Patch(facecolor=self.COLORS['Beta_Liquid'], label='Beta (Shell/Back): Liquid'),

                Line2D([0], [0], marker='o', color='w', label='Skin A (Ag-rich)',
                       markerfacecolor='white', markeredgecolor=self.COLORS['Skin_A'], markeredgewidth=0.5, markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Skin B (Cu-rich)',
                       markerfacecolor='white', markeredgecolor=self.COLORS['Skin_B'], markeredgewidth=0.5, markersize=10),
                Line2D([0], [0], marker=self._create_segment_marker(0.3), color='w', label='Janus: Alpha (Segment) / Beta (Back)',
                       markerfacecolor='gray', markeredgecolor='black', markersize=12),
            ]
            
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title="Legend")
            plt.tight_layout()
            
            # Save figure if save_dir is provided
            if self.save_dir:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                if self.timestamp:
                    filename = f"PhaseDiagram_n{n:.1e}_{self.timestamp}.png"
                else:
                    filename = f"PhaseDiagram_n{n:.1e}.png"
                filepath = os.path.join(self.save_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
            
            plt.show()

    def _get_phase_colors(self, phase_series, role):
        # role is "Alpha" or "Beta"
        return phase_series.map({
            "FCC": self.COLORS[f"{role}_FCC"],
            "Liquid": self.COLORS[f"{role}_Liquid"]
        })

    def _get_skin_styles(self, df):
        # Returns (edge_colors, line_widths)
        # Default (No Skin)
        edge_colors = pd.Series(['none'] * len(df), index=df.index)
        line_widths = pd.Series([0.0] * len(df), index=df.index)
        
        has_skin = df["HasSkin"]
        
        if has_skin.any():
            # Use a fixed width for the skin outline to avoid visual overflow
            # 1.5 is a reasonable thickness for the base_size of 35
            FIXED_WIDTH = 0.5

            # Skin A (Red)
            mask_a = has_skin & (df["xB_skin"] < 0.5)
            edge_colors[mask_a] = self.COLORS["Skin_A"]
            line_widths[mask_a] = FIXED_WIDTH
            
            # Skin B (Blue)
            mask_b = has_skin & (df["xB_skin"] >= 0.5)
            edge_colors[mask_b] = self.COLORS["Skin_B"]
            line_widths[mask_b] = FIXED_WIDTH
            
        return edge_colors, line_widths

    def _create_segment_marker(self, ratio):
        # Create a polygon for a circular segment representing the Alpha fraction
        # x goes from -1 to 1. x = 2*ratio - 1 approximates the chord position.
        x_cut = np.clip(2 * ratio - 1, -0.99, 0.99)
        
        # Angles for the arc (Left side of the circle)
        theta_start = np.arccos(x_cut)
        theta_end = 2 * np.pi - theta_start
        
        angles = np.linspace(theta_start, theta_end, 30)
        xs = np.cos(angles)
        ys = np.sin(angles)
        
        verts = np.column_stack([xs, ys])
        return verts

if __name__ == "__main__":
    # Automatically find the latest result file in the Results folder
    list_of_files = glob.glob('Results/*.csv') 
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        PhaseDiagramPlotting3Phase(latest_file)
    else:
        print("No result files found in Results/ directory.")
