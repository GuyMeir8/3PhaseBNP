import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


class PhaseDiagramPlotting:
    def __init__(self, file_name):
        # 1. Define Discrete Colors for Non-Gradient Geometries
        # 1. Define Discrete Colors for Non-Gradient Geometries
        # 1. Define Discrete Colors for Non-Gradient Geometries
        self.PHASE_COLORS = {
            # --- Single Phase ---
            "SINGLE_PHASE__FCC": "#1f77b4",
            "SINGLE_PHASE__Liquid": "#ff7f0e",

            # --- Core Shell (Gradient Fallbacks) ---
            "CORE_SHELL__FCC+FCC": "#d62728",
            "CORE_SHELL__FCC+Liquid": "#9467bd",
            "CORE_SHELL__Liquid+FCC": "#c5b0d5",

            # --- Janus (Standard) ---
            "JANUS__FCC+FCC": "#2ca02c",
            "JANUS__FCC+Liquid": "#8c564b",
            "JANUS__Liquid+FCC": "#e377c2",

            # --- Spheric Janus (Standard) ---
            "SPHERIC_JANUS__FCC+FCC": "#2ca02c",
            "SPHERIC_JANUS__FCC+Liquid": "#8c564b",
            "SPHERIC_JANUS__Liquid+FCC": "#e377c2",

            # ____________________ SKIN A (Ag-Rich) ____________________ #

            # CoreShell + Skin A
            "CORE_SHELL+SkinA__FCC+FCC": "#00CED1",
            "CORE_SHELL+SkinA__FCC+Liquid": "#20B2AA",
            "CORE_SHELL+SkinA__Liquid+FCC": "#AFEEEE",

            # SinglePhase + Skin A
            "SINGLE_PHASE+SkinA__FCC": "#000080",
            "SINGLE_PHASE+SkinA__Liquid": "#4169E1",

            # Janus + Skin A
            "JANUS+SkinA__FCC+FCC": "#32cd32",
            "JANUS+SkinA__FCC+Liquid": "#006400",
            "JANUS+SkinA__Liquid+FCC": "#90ee90",

            # Spheric Janus + Skin A (COMPLETE)
            "SPHERIC_JANUS+SkinA__FCC+FCC": "#32cd32",
            "SPHERIC_JANUS+SkinA__FCC+Liquid": "#006400",  # <--- Added
            "SPHERIC_JANUS+SkinA__Liquid+FCC": "#90ee90",  # <--- Added

            # ____________________ SKIN B (Cu-Rich) ____________________ #

            # CoreShell + Skin B
            "CORE_SHELL+SkinB__FCC+FCC": "#BCBD22",
            "CORE_SHELL+SkinB__FCC+Liquid": "#808000",
            "CORE_SHELL+SkinB__Liquid+FCC": "#EEE8AA",

            # SinglePhase + Skin B
            "SINGLE_PHASE+SkinB__FCC": "#D2691E",
            "SINGLE_PHASE+SkinB__Liquid": "#FF4500",

            # Janus + Skin B
            "JANUS+SkinB__FCC+FCC": "#ff00ff",
            "JANUS+SkinB__FCC+Liquid": "#800080",
            "JANUS+SkinB__Liquid+FCC": "#dda0dd",

            # Spheric Janus + Skin B (COMPLETE)
            "SPHERIC_JANUS+SkinB__FCC+FCC": "#ff00ff",
            "SPHERIC_JANUS+SkinB__FCC+Liquid": "#800080",  # <--- Added
            "SPHERIC_JANUS+SkinB__Liquid+FCC": "#dda0dd",  # <--- Added
        }

        # 2. Load and Clean Data
        self.df = pd.read_csv(file_name)

        rename_map = {
            "xB_total": "x_B_total",
            "G_min": "G_total",
            "GeoType": "geometry",
            "PhaseAlpha": "alpha_phase",
            "PhaseBeta": "beta_phase",
            "OuterShellType": "outer_shell"
        }
        self.df.rename(columns={k: v for k, v in rename_map.items() if k in self.df.columns}, inplace=True)

        for col in ["geometry", "outer_shell", "alpha_phase", "beta_phase"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()

        # Ensure numeric
        for c in ["x_B_beta", "n_alpha", "n_total", "x_B_total", "T", "G_total"]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

        # ---------------------------------------------------------
        # 3. FILTERS (Phantom Core & Pure Shell)
        # ---------------------------------------------------------
        print("--- DEBUG: Filter Diagnostics ---")
        is_coreshell_geo = self.df["geometry"].str.upper().eq("CORE_SHELL")

        # Filter 1: Pure Shells (Treat as Skin Geometry)
        SHELL_PURITY_LIMIT = 10e-3
        is_pure_shell = (self.df["x_B_beta"] < SHELL_PURITY_LIMIT) | (self.df["x_B_beta"] > (1.0 - SHELL_PURITY_LIMIT))

        # Filter 2: Phantom Cores (Treat as Single Phase)
        # CORE_SIZE_LIMIT = 0.01
        # core_fraction = (self.df["n_alpha"] / self.df["n_total"]).fillna(1.0)
        # is_phantom_core = core_fraction < CORE_SIZE_LIMIT

        # mask_remove = is_coreshell_geo & (is_pure_shell | is_phantom_core)
        mask_remove = is_coreshell_geo & is_pure_shell
        print(f"Removing {mask_remove.sum()} artifact rows.")
        self.df = self.df[~mask_remove].copy()

        # 4. Find Minimum Energy
        self.df_min = self.df.loc[self.df.groupby(["n_total", "x_B_total", "T"])["G_total"].idxmin()].copy()

        # 5. Clean Dust
        self.df_min = self.df_min.apply(self.enforce_phase_threshold, axis=1)

        # 6. Generate Labels (Standard, no size tags)
        self.df_min["phase_label"] = self.df_min.apply(self.make_label, axis=1)

        # 7. Smoothing (Speckle Removal)
        cleaned_dfs = []
        for n in self.df_min["n_total"].unique():
            subset = self.df_min[self.df_min["n_total"] == n].copy()
            cleaned_subset = self.remove_speckles_from_slice(subset, passes=100)
            cleaned_dfs.append(cleaned_subset)
        self.df_min = pd.concat(cleaned_dfs)

        # 8. Plot with Gradients
        self.plot_all_phase_diagrams_per_n_total()

    def plot_all_phase_diagrams_per_n_total(self):
        df_plot = self.df_min

        for n in df_plot["n_total"].unique():
            subset = df_plot[df_plot["n_total"] == n]

            plt.figure(figsize=(11, 7))

            # --- SPLIT DATA: Gradient vs Discrete ---
            # Gradient = Standard Core-Shell (No Skin)
            # Discrete = Everything else

            # Helper to check for "No Skin"
            # make_label adds "+Skin" if skin exists. If "CORE_SHELL__" is in label but "+Skin" isn't...
            # A safer way: Check geometry column and outer_shell column

            # Identify Standard Core-Shell rows
            is_standard_cs = (subset["geometry"] == "CORE_SHELL") & \
                             (~subset["phase_label"].str.contains("Skin"))

            subset_gradient = subset[is_standard_cs].copy()
            subset_discrete = subset[~is_standard_cs].copy()

            # --- LAYER 1: Plot Discrete Points (Janus, Skin, Single) ---
            if not subset_discrete.empty:
                sns.scatterplot(
                    data=subset_discrete,
                    x="x_B_total", y="T",
                    hue="phase_label",
                    palette=self.PHASE_COLORS,
                    alpha=1.0, s=25, linewidth=0,
                    zorder=10  # Draw on top
                )

            # --- LAYER 2: Plot Gradient Points (Core-Shell) ---
            # We calculate "Shell Fraction" (1 - n_alpha/n_total)
            # 0.0 (Pure Core) -> Light Color
            # 1.0 (Pure Shell) -> Dark Color

            if not subset_gradient.empty:
                subset_gradient["core_frac"] = subset_gradient["n_alpha"] / subset_gradient["n_total"]
                subset_gradient["shell_frac"] = 1.0 - subset_gradient["core_frac"]

                # Split by Phase Pair to assign different Color Maps

                # 1. FCC + FCC (Red Gradient)
                mask_ff = (subset_gradient["alpha_phase"] == "FCC") & (subset_gradient["beta_phase"] == "FCC")
                self._plot_gradient_scatter(subset_gradient[mask_ff], "Reds", "FCC+FCC")

                # 2. FCC + Liquid (Purple Gradient)
                mask_fl = (subset_gradient["alpha_phase"] == "FCC") & (subset_gradient["beta_phase"] == "Liquid")
                self._plot_gradient_scatter(subset_gradient[mask_fl], "Purples", "FCC+Liq")

                # 3. Liquid + FCC (Blue-Purple Gradient)
                mask_lf = (subset_gradient["alpha_phase"] == "Liquid") & (subset_gradient["beta_phase"] == "FCC")
                self._plot_gradient_scatter(subset_gradient[mask_lf], "BuPu", "Liq+FCC")

            plt.title(f"Phase Diagram (n={n:.1e})\nSmooth Gradient = Shell Thickness (Darker = Thicker Shell)")
            plt.xlabel("x_B_total")
            plt.ylabel("Temperature [K]")

            # Move legend outside
            plt.legend(title="Geometries", loc='upper left', bbox_to_anchor=(1, 1), markerscale=1.5)
            plt.tight_layout()
            plt.show()

    def _plot_gradient_scatter(self, df, cmap_name, label_prefix):
        if df.empty: return

        # Create truncated colormap so we don't get invisible white points
        # We take the range 0.3 (Light) to 1.0 (Dark) of the standard map
        base_cmap = plt.get_cmap(cmap_name)
        color_range = mcolors.LinearSegmentedColormap.from_list(
            f"trunc_{cmap_name}", base_cmap(np.linspace(0.3, 1.0, 100))
        )

        plt.scatter(
            df["x_B_total"], df["T"],
            c=df["shell_frac"],
            cmap=color_range,
            vmin=0.0, vmax=1.0,
            s=25, marker='s', edgecolors='none',
            alpha=0.8,
            label=f"CoreShell {label_prefix} (Gradient)",
            zorder=1  # Draw behind discrete
        )

    def enforce_phase_threshold(self, row):
        THRESHOLD = 0.001
        if "n_alpha" not in row: return row
        n_tot = row["n_total"]
        if n_tot == 0: return row

        if "SINGLE_PHASE" in row["geometry"]: return row

        na, nb = row["n_alpha"], row["n_beta"]

        if (na / n_tot) < THRESHOLD:
            row["geometry"] = "SINGLE_PHASE"
            row["alpha_phase"] = row["beta_phase"]
            return row
        if (nb / n_tot) < THRESHOLD:
            row["geometry"] = "SINGLE_PHASE"
            return row
        return row

    def remove_speckles_from_slice(self, df_slice, passes=1):
        grid_df = df_slice.pivot(index="T", columns="x_B_total", values="phase_label")
        grid = grid_df.values
        rows, cols = grid.shape
        for i in range(passes):
            new_grid = grid.copy()
            changes_made = 0
            for r in range(rows):
                for c in range(cols):
                    current_val = grid[r, c]
                    valid_neighbors = []
                    neighbors_matching_self = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                val = grid[nr, nc]
                                if pd.notna(val):
                                    valid_neighbors.append(val)
                                    if val == current_val: neighbors_matching_self += 1
                    if not valid_neighbors: continue
                    is_edge = (r == 0) or (r == rows - 1) or (c == 0) or (c == cols - 1)
                    limit = 0 if is_edge else 1
                    if neighbors_matching_self <= limit:
                        counts = {}
                        for n in valid_neighbors: counts[n] = counts.get(n, 0) + 1
                        if counts:
                            most = max(counts, key=counts.get)
                            if most != current_val:
                                new_grid[r, c] = most
                                changes_made += 1
            grid = new_grid
            if changes_made == 0: break

        cleaned = pd.DataFrame(grid, index=grid_df.index, columns=grid_df.columns).reset_index().melt(id_vars="T",
                                                                                                      value_name="clean")
        df_slice = pd.merge(df_slice, cleaned, on=["T", "x_B_total"], how="left")
        df_slice["phase_label"] = df_slice["clean"]
        return df_slice

    @staticmethod
    def make_label(row):
        geo = row["geometry"]
        a, b = row["alpha_phase"], row["beta_phase"]
        shell = str(row.get("outer_shell", "None"))

        suffix = ""
        if shell not in ["None", "nan", "NaN", ""]:
            if "_Ag" in shell:
                suffix = "+SkinA"
            elif "_Cu" in shell:
                suffix = "+SkinB"
            else:
                suffix = "+Skin"

        if "SINGLE_PHASE" in geo: return f"{geo}{suffix}__{a}"
        return f"{geo}{suffix}__{a}+{b}"


if __name__ == "__main__":
    pass