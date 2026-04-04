import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import os
import datetime

class PhaseDiagramPlotting3Phase:
    def __init__(self, file_name, save_dir=None, timestamp=None, plot_title_suffix="", auto_show=True):
        self.auto_show = auto_show
        self.plot_title_suffix = plot_title_suffix
        self.file_name = file_name
        
        if save_dir is None:
            self.save_dir = "Results"
        else:
            self.save_dir = save_dir

        if timestamp is None:
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.timestamp = timestamp
        
        # 1. Define Colors
        self.COLORS = {
            "Single_Liquid": "lightblue",
            "Single_FCC": "#FFDB58", 
            "Alpha_FCC": "lightgreen",
            "Beta_FCC": "darkgreen",
            "Alpha_Liquid": "#B57EDC", 
            "Beta_Liquid": "darkviolet",
            "Skin_A": "red",
            "Skin_B": "blue"
        }

        # 2. Load Data
        self.df_raw = pd.read_csv(file_name)
        self.df = self.df_raw.copy()
        self.preprocess_data()

        # 3. Filter Valid Results
        self.df = self.df[(self.df["G_min"] != 1.0) & (~np.isinf(self.df["G_min"]))].copy()
        
        if self.df.empty:
            print("No valid results found to plot.")
            return

        # 4. Find Minimum Energy Configuration per Point
        idx = self.df.groupby(["n_total", "xB_total", "T"])["G_min"].idxmin()
        self.df_min = self.df.loc[idx].copy()

        # 5. Plot
        self.plot_all_phase_diagrams_per_n_total()

    def preprocess_data(self):
        for col in ["Geometry", "PhaseAlpha", "PhaseBeta"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
                
        if "HasSkin" in self.df.columns:
            self.df["HasSkin"] = self.df["HasSkin"].replace({'True': True, 'False': False, '1': True, '0': False})
            self.df["HasSkin"] = self.df["HasSkin"].fillna(False).astype(bool)
            
        if "Is_Suspect" in self.df.columns:
            self.df["Is_Suspect"] = self.df["Is_Suspect"].replace({'True': True, 'False': False, '1': True, '0': False})
            self.df["Is_Suspect"] = self.df["Is_Suspect"].fillna(False).astype(bool)
        
        cols = ["xB_total", "T", "G_min", "n_total", "n_alpha", "n_beta", "xB_skin"]
        for c in cols:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

    def plot_all_phase_diagrams_per_n_total(self):
        df_plot = self.df_min
        
        for n in df_plot["n_total"].unique():
            subset = df_plot[df_plot["n_total"] == n]
            if subset.empty: continue
            
            fig = plt.figure(figsize=(12, 8))
            base_size = 30 
            MIN_VISIBLE_RATIO = 0.2 

            # 1. Single Phase
            df_single = subset[subset["Geometry"] == "SinglePhase"].copy()
            if not df_single.empty:
                colors = df_single["PhaseAlpha"].map({
                    "Liquid": self.COLORS["Single_Liquid"],
                    "FCC": self.COLORS["Single_FCC"]
                })
                plt.scatter(df_single["xB_total"], df_single["T"], c=colors, edgecolors='none', s=base_size, marker='o', label="Single Phase", zorder=1)

            # 2. Core-Shell
            df_cs = subset[subset["Geometry"] == "Core_Shell"].copy()
            if not df_cs.empty:
                df_cs["alpha_ratio"] = df_cs["n_alpha"] / df_cs["n_total"]
                df_cs["alpha_ratio"] = df_cs["alpha_ratio"].fillna(0)
                
                plot_alpha_ratio = df_cs["alpha_ratio"].clip(lower=MIN_VISIBLE_RATIO, upper=1.0 - MIN_VISIBLE_RATIO)
                c_alpha = self._get_phase_colors(df_cs["PhaseAlpha"], "Alpha")
                c_beta = self._get_phase_colors(df_cs["PhaseBeta"], "Beta")

                plt.scatter(df_cs["xB_total"], df_cs["T"], c=c_beta, edgecolors='none', s=base_size, marker='o', zorder=1)
                s_inner = base_size * plot_alpha_ratio
                plt.scatter(df_cs["xB_total"], df_cs["T"], c=c_alpha, edgecolors='none', s=s_inner, marker='o', label="Core-Shell", zorder=2)

            # 3. Janus
            df_janus = subset[subset["Geometry"] == "Janus"].copy()
            if not df_janus.empty:
                df_janus["alpha_ratio"] = df_janus["n_alpha"] / df_janus["n_total"]
                df_janus["alpha_ratio"] = df_janus["alpha_ratio"].fillna(0)
                
                plot_alpha_ratio = df_janus["alpha_ratio"].clip(lower=MIN_VISIBLE_RATIO, upper=1.0 - MIN_VISIBLE_RATIO)
                c_alpha = self._get_phase_colors(df_janus["PhaseAlpha"], "Alpha")
                c_beta = self._get_phase_colors(df_janus["PhaseBeta"], "Beta")

                plt.scatter(df_janus["xB_total"], df_janus["T"], c=c_beta, edgecolors='none', s=base_size, marker='o', zorder=1)
                for i in range(len(df_janus)):
                    ratio = plot_alpha_ratio.iloc[i]
                    verts = self._create_segment_marker(ratio)
                    plt.scatter(
                        df_janus["xB_total"].iloc[i], df_janus["T"].iloc[i],
                        c=[c_alpha.iloc[i]], edgecolors='none',
                        s=base_size, marker=verts, zorder=2
                    )

            # 4. Skin Overlay Layer
            df_skin = subset[subset["HasSkin"] == True].copy()
            if not df_skin.empty:
                skin_colors = np.where(df_skin["xB_skin"] < 0.5, self.COLORS["Skin_A"], self.COLORS["Skin_B"])
                plt.scatter(df_skin["xB_total"], df_skin["T"], facecolors='none', edgecolors=skin_colors, s=base_size, linewidths=0.5, zorder=10)
                    
            title = f"Phase Diagram (n={n:.1e})"
            if self.plot_title_suffix:
                title += f" {self.plot_title_suffix}"
            plt.title(title)
            plt.xlabel("xB_total (Composition)")
            plt.ylabel("Temperature [K]")
            
            # Custom Legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Single Phase (Liquid)', markerfacecolor=self.COLORS['Single_Liquid'], markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Single Phase (FCC)', markerfacecolor=self.COLORS['Single_FCC'], markersize=10),
                Patch(facecolor=self.COLORS['Alpha_FCC'], label='Alpha (Core/Seg): FCC'),
                Patch(facecolor=self.COLORS['Beta_FCC'], label='Beta (Shell/Back): FCC'),
                Patch(facecolor=self.COLORS['Alpha_Liquid'], label='Alpha (Core/Seg): Liquid'),
                Patch(facecolor=self.COLORS['Beta_Liquid'], label='Beta (Shell/Back): Liquid'),
                Line2D([0], [0], marker='o', color='w', label='Skin A (Ag-rich)', markerfacecolor='none', markeredgecolor=self.COLORS['Skin_A'], markeredgewidth=0.5, markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Skin B (Cu-rich)', markerfacecolor='none', markeredgecolor=self.COLORS['Skin_B'], markeredgewidth=0.5, markersize=10),
                Line2D([0], [0], marker=self._create_segment_marker(0.3), color='w', label='Janus: Alpha (Segment) / Beta (Back)', markerfacecolor='gray', markeredgecolor='black', markersize=12),
            ]
            
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title="Legend")
            plt.tight_layout()
            
            if self.save_dir:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                base_name = os.path.splitext(os.path.basename(self.file_name))[0]
                filename = f"{base_name}_n{n:.1e}.png"
                filepath = os.path.join(self.save_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
            
            if self.auto_show:
                plt.show()
            else:
                plt.close(fig)

    def _get_phase_colors(self, phase_series, role):
        return phase_series.map({"FCC": self.COLORS[f"{role}_FCC"], "Liquid": self.COLORS[f"{role}_Liquid"]})

    def _create_segment_marker(self, ratio):
        x_cut = np.clip(2 * ratio - 1, -0.99, 0.99)
        theta_start = np.arccos(x_cut)
        theta_end = 2 * np.pi - theta_start
        angles = np.linspace(theta_start, theta_end, 30)
        xs = np.cos(angles)
        ys = np.sin(angles)
        verts = np.column_stack([xs, ys])
        return verts

if __name__ == "__main__":
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "Results")
    
    print("Opening file dialog to select the data file manually...")
    target_file = None
    
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.attributes('-topmost', True) 
        root.withdraw() 
        target_file = filedialog.askopenfilename(
            initialdir=target_dir if os.path.exists(target_dir) else script_dir,
            title="Select raw data CSV file for 2D Phase Diagram",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
    except Exception as e:
        print(f"Could not open file dialog: {e}")
        import glob
        list_of_files = glob.glob(os.path.join(target_dir, '*.csv'))
        if list_of_files:
            target_file = max(list_of_files, key=os.path.getctime)
            print(f"Fallback: using latest file: {target_file}")
            
    if target_file:
        print(f"Using file: {target_file}")
        PhaseDiagramPlotting3Phase(target_file)
    else:
        print("No file selected. Exiting.")