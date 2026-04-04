import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import filedialog

class PhaseSlicePlotter:
    def __init__(self, file_name, independent_var, constant_val, n_total=None, save_dir="Results", show_transitions=True):
        """
        Generates 1D slice plots of phase fractions and compositions.
        Expects a PRE-CLEANED csv file. Does no data manipulation.
        """
        self.file_name = file_name
        self.independent_var = independent_var.lower()
        self.constant_val = constant_val
        self.save_dir = save_dir
        self.show_transitions = show_transitions
        
        self.df = pd.read_csv(file_name)
        
        # 1. Ensure minimal types
        for col in ["Geometry", "PhaseAlpha", "PhaseBeta"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
        if "HasSkin" in self.df.columns:
            self.df["HasSkin"] = self.df["HasSkin"].replace({'True': True, 'False': False, '1': True, '0': False}).fillna(False).astype(bool)
                
        cols_numeric = ["xB_total", "T", "G_min", "n_total", "n_alpha", "n_beta", "xB_alpha", "xB_beta", "xB_skin"]
        for c in cols_numeric:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')
                
        # 2. Filter to minimum energy states (if file hasn't already been reduced)
        self.df = self.df[(self.df["G_min"] != 1.0) & (~np.isinf(self.df["G_min"]))].copy()
        if self.df.empty:
            print("No valid results found in file.")
            return

        idx = self.df.groupby(["n_total", "xB_total", "T"])["G_min"].idxmin()
        self.df_min = self.df.loc[idx].copy()
        
        if n_total is None:
            self.n_total = self.df_min["n_total"].unique()[0]
        else:
            self.n_total = n_total
            
        self.df_plot = self.df_min[self.df_min["n_total"] == self.n_total].copy()
        
        # 3. Filter to the specific 1D slice (Isotherm or Isopleth)
        if self.independent_var in ['temperature', 't']:
            self.x_col = 'T'
            self.const_col = 'xB_total'
            self._filter_to_constant()
            self.title_prefix = f"Isopleth (xB_total = {self.constant_val})"
            self.x_label = "Temperature [K]"
        elif self.independent_var in ['composition', 'xb', 'xb_total']:
            self.x_col = 'xB_total'
            self.const_col = 'T'
            self._filter_to_constant()
            self.title_prefix = f"Isotherm (T = {self.constant_val} K)"
            self.x_label = "Composition (xB_total)"
        else:
            raise ValueError("independent_var must be 'temperature' or 'composition'")
            
        self.df_plot = self.df_plot.sort_values(by=self.x_col).reset_index(drop=True)
        
        # 4. Format the final visual data
        self.prepare_visual_data()
        self.plot()
        
    def _filter_to_constant(self):
        available_consts = self.df_plot[self.const_col].unique()
        if len(available_consts) == 0:
            print(f"No data available for n_total={self.n_total}.")
            self.df_plot = pd.DataFrame()
            return
            
        closest_const = available_consts[np.argmin(np.abs(available_consts - self.constant_val))]
        if abs(closest_const - self.constant_val) > 1e-3:
            print(f"Warning: Exact {self.const_col}={self.constant_val} not found. Using closest available: {closest_const}")
        self.constant_val = closest_const
        self.df_plot = self.df_plot[abs(self.df_plot[self.const_col] - self.constant_val) < 1e-4]

    def prepare_visual_data(self):
        """Converts moles to fractions strictly for the Y-axis of the plot."""
        if self.df_plot.empty: return
        
        nt = self.df_plot['n_total']
        self.df_plot['f_alpha'] = self.df_plot['n_alpha'] / nt
        self.df_plot['f_beta'] = self.df_plot['n_beta'] / nt
        
        # Calculate skin fraction if skin exists
        self.df_plot['f_skin'] = np.where(
            self.df_plot['HasSkin'], 
            (nt - self.df_plot['n_alpha'].fillna(0) - self.df_plot['n_beta'].fillna(0)) / nt, 
            np.nan
        )

        # Create strings for the vertical transition markers
        def get_state(row):
            geo = row["Geometry"]
            if geo == "SinglePhase":
                return f"SinglePhase ({row['PhaseAlpha']})"
            else:
                skin_str = "+Skin" if row['HasSkin'] else ""
                return f"{geo} ({row['PhaseAlpha']}/{row['PhaseBeta']}){skin_str}"
                
        self.df_plot['State'] = self.df_plot.apply(get_state, axis=1)
        
    def plot(self):
        if self.df_plot.empty:
            print("No data available to plot for these parameters.")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        x = self.df_plot[self.x_col].values
        
        # --- Top Plot: Phase Fractions ---
        ax1.plot(x, self.df_plot['f_alpha'], label='n_alpha / n_total', color='#2ca02c', linewidth=2.5, marker='o', markersize=4)
        ax1.plot(x, self.df_plot['f_beta'], label='n_beta / n_total', color='#1f77b4', linewidth=2.5, marker='s', markersize=4)
        ax1.plot(x, self.df_plot['f_skin'], label='n_skin / n_total', color='#d62728', linewidth=2.5, marker='^', markersize=4)
        
        ax1.set_ylabel("Mole Fraction")
        ax1.set_title(f"{self.title_prefix}, n = {self.n_total:.1e}\nPhase Fractions", fontsize=14, pad=15)
        ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_ylim(-0.05, 1.05)
        
        # --- Bottom Plot: Compositions ---
        ax2.plot(x, self.df_plot['xB_alpha'], label='xB_alpha', color='#2ca02c', linewidth=2, marker='o', markersize=5)
        ax2.plot(x, self.df_plot['xB_beta'], label='xB_beta', color='#1f77b4', linewidth=2, marker='s', markersize=5)
        ax2.plot(x, self.df_plot['xB_skin'], label='xB_skin', color='#d62728', linewidth=2, marker='^', markersize=5)
        
        ax2.set_xlabel(self.x_label, fontsize=12)
        ax2.set_ylabel("Composition (xB)", fontsize=12)
        ax2.set_title("Phase Compositions", fontsize=14, pad=10)
        ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_ylim(-0.05, 1.05)
        
        # --- Add State Change Markers ---
        if self.show_transitions:
            states = self.df_plot['State'].values
            text_heights = [0.8, 0.5, 0.2] 
            text_idx = 0
            
            for i in range(1, len(states)):
               if states[i] != states[i-1]:
                    change_x = (x[i] + x[i-1]) / 2.0
                    
                    ax1.axvline(x=change_x, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
                    ax2.axvline(x=change_x, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
                    
                    trans_text = f"{states[i-1]} \n-> {states[i]}"
                    height = text_heights[text_idx % len(text_heights)]
                    text_idx += 1
                    
                    ax1.text(change_x, height, trans_text, rotation=90, va='center', ha='right', 
                             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'),
                             fontsize=9)
                         
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        
        # Save figure
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            
            base_name = os.path.splitext(os.path.basename(self.file_name))[0]
            var_name = "T" if self.independent_var in ['temperature', 't'] else "xB"
            filename = f"{base_name}_1D_{var_name}{self.constant_val}_n{self.n_total:.1e}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
            
        plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Opening file dialog to select the CLEANED data file...")
    target_file = None
    try:
        root = tk.Tk()
        root.attributes('-topmost', True) 
        root.withdraw() 
        target_file = filedialog.askopenfilename(
            initialdir=script_dir,
            title="Select CLEANED data CSV file",
            filetypes=(("CSV files", "*cleaned.csv"), ("All files", "*.*"))
        )
    except Exception as e:
        print(f"Could not open file dialog: {e}")
            
    if target_file:
        print(f"Using file: {target_file}")
        
        # Isopleth (Constant Composition of 0.98, Sweeping Temperature)
        PhaseSlicePlotter(
            file_name=target_file, 
            independent_var='temperature', 
            constant_val=0.98, 
            show_transitions=True
        )
        
    else:
        print("No file selected or found. Exiting.")