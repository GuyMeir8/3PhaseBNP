import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

class PhaseSlicePlotter:
    def __init__(self, file_name, independent_var, constant_val, n_total=None, save_dir="Results"):
        """
        Generates 1D slice plots of phase fractions and compositions.
        
        Args:
            file_name: Path to the raw results CSV file.
            independent_var: 'temperature' or 'composition'. This determines the x-axis.
            constant_val: The value of the OTHER variable to hold constant.
                          (e.g. if independent_var='temperature', constant_val is the xB_total).
            n_total: Specific mole amount to plot. Defaults to the first one found if None.
        """
        self.file_name = file_name
        self.independent_var = independent_var.lower()
        self.constant_val = constant_val
        self.save_dir = save_dir
        
        self.df = pd.read_csv(file_name)
        
        # Clean up strings
        for col in ["Geometry", "PhaseAlpha", "PhaseBeta"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
                
        if "HasSkin" in self.df.columns:
            # Handle potential string conversions of bools
            self.df["HasSkin"] = self.df["HasSkin"].replace({'True': True, 'False': False, '1': True, '0': False})
            self.df["HasSkin"] = self.df["HasSkin"].fillna(False).astype(bool)
                
        # Ensure numeric
        cols_numeric = ["xB_total", "T", "G_min", "n_total", "n_alpha", "n_beta", "xB_alpha", "xB_beta", "xB_skin"]
        for c in cols_numeric:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')
                
        # Filter valid results
        self.df = self.df[(self.df["G_min"] != 1.0) & (~np.isinf(self.df["G_min"]))].copy()
        
        if self.df.empty:
            print("No valid results found in file.")
            return

        # Get minimum energy configs
        idx = self.df.groupby(["n_total", "xB_total", "T"])["G_min"].idxmin()
        self.df_min = self.df.loc[idx].copy()
        
        if n_total is None:
            self.n_total = self.df_min["n_total"].unique()[0]
        else:
            self.n_total = n_total
            
        self.df_plot = self.df_min[self.df_min["n_total"] == self.n_total].copy()
        
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
            
        self.df_plot = self.df_plot.sort_values(by=self.x_col)
        
        self.prepare_plot_data()
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

    def prepare_plot_data(self):
        if self.df_plot.empty: return
        
        # Calculate derived fractions
        def calc_fractions(row):
            nt = row['n_total']
            
            # SinglePhase usually has NaN for n_beta and n_alpha might just be n_total
            if row['Geometry'] == 'SinglePhase' or pd.isna(row['n_alpha']):
                na = nt
                nb = 0.0
            else:
                na = row['n_alpha']
                nb = row['n_beta'] if not pd.isna(row['n_beta']) else 0.0
                
            has_skin = row['HasSkin']
            ns = nt - na - nb if has_skin else 0.0
            
            # Prevent tiny negative values due to float precision
            na = max(0.0, na)
            nb = max(0.0, nb)
            ns = max(0.0, ns)
            
            f_a = na / nt
            f_b = nb / nt
            f_s = ns / nt
            
            return pd.Series([f_a, f_b, f_s])
            
        self.df_plot[['f_alpha', 'f_beta', 'f_skin']] = self.df_plot.apply(calc_fractions, axis=1)
        
        # Ensure compositions correctly show empty arrays when they don't apply
        def clean_compositions(row):
            if row['Geometry'] == 'SinglePhase':
                row['xB_beta'] = np.nan
            if not row['HasSkin']:
                row['xB_skin'] = np.nan
            return row
            
        self.df_plot = self.df_plot.apply(clean_compositions, axis=1)
        
        # State label for transitions
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
        xb_a = self.df_plot['xB_alpha'].values
        xb_b = self.df_plot['xB_beta'].values
        xb_s = self.df_plot['xB_skin'].values
        
        ax2.plot(x, xb_a, label='xB_alpha', color='#2ca02c', linewidth=2, marker='o', markersize=5)
        ax2.plot(x, xb_b, label='xB_beta', color='#1f77b4', linewidth=2, marker='s', markersize=5)
        ax2.plot(x, xb_s, label='xB_skin', color='#d62728', linewidth=2, marker='^', markersize=5)
        
        ax2.set_xlabel(self.x_label, fontsize=12)
        ax2.set_ylabel("Composition (xB)", fontsize=12)
        ax2.set_title("Phase Compositions", fontsize=14, pad=10)
        ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_ylim(-0.05, 1.05)
        
        # --- Add State Change Markers ---
        states = self.df_plot['State'].values
        text_heights = [0.8, 0.5, 0.2] # Alternating heights so text doesn't overlap on clustered transitions
        text_idx = 0
        
        for i in range(1, len(states)):
           if states[i] != states[i-1]:
                change_x = (x[i] + x[i-1]) / 2.0
                
                # Draw vertical lines
                ax1.axvline(x=change_x, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
                ax2.axvline(x=change_x, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
                
                # Annotate the transition on the top plot
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
    # Use an absolute path based on the script's location to avoid working directory issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "Results_worth_saving")
    search_pattern = os.path.join(target_dir, "*.csv")
    
    list_of_files = glob.glob(search_pattern)
    list_of_files = [f for f in list_of_files if 'checkpoint' not in f]
    
    target_file = None
    if list_of_files:
        # Sort to get the first file deterministically
        target_file = sorted(list_of_files)[0]
    else:
        print(f"No result files found in directory: {target_dir}")
        print("Opening file dialog to select the data file manually...")
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.attributes('-topmost', True) # Bring the dialog to the front
            root.withdraw() # Hide the main empty Tkinter window
            target_file = filedialog.askopenfilename(
                initialdir=script_dir,
                title="Select raw data CSV file",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
            )
        except Exception as e:
            print(f"Could not open file dialog: {e}")
            
    if target_file:
        print(f"Using file: {target_file}")
        
        # Isotherm (Constant Temperature of 1100K, Sweeping Composition)
        PhaseSlicePlotter(target_file, independent_var='composition', constant_val=1100.0)
        
    else:
        print("No file selected or found. Exiting.")