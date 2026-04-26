import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def find_melting_transitions():
    print("Opening file dialog to select the AgCu RAW CSV file(s)...")
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Select RAW Data CSV Files (Important!)",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )

    if not file_paths:
        print("No files selected. Exiting.")
        return

    target_xBs = [0.01, 0.99]
    
    for file_path in file_paths:
        print(f"\n--- Results for {os.path.basename(file_path)} ---")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file: {e}")
            continue

        if not all(col in df.columns for col in ['n_total', 'xB_total', 'T', 'Geometry', 'PhaseAlpha']):
            print("File missing required columns.")
            continue

        n_values_in_file = df['n_total'].unique()
        if len(n_values_in_file) == 0:
            print("Could not find any 'n_total' values in the file. Skipping.")
            continue

        if len(n_values_in_file) > 1:
            print(f"Info: Multiple n_total values found: {n_values_in_file}. Results will be shown for each.")

        for n in n_values_in_file:
            print(f"\nProcessing for n = {n:g}:")
            df_n = df[df['n_total'] == n]

            for xb in target_xBs:
                df_xb = df_n[np.isclose(df_n['xB_total'], xb, atol=1e-4)]
                if df_xb.empty:
                    continue
                    
                if 'G_min' not in df_xb.columns:
                    print(f"  For xB={xb:g}, 'G_min' column missing.")
                    continue

                # Filter to only SinglePhase, and ignore failed calculations (G_min >= 1.0)
                df_sp = df_xb[(df_xb['Geometry'].astype(str).str.strip() == 'SinglePhase') & (df_xb['G_min'] < 1.0)].copy()
                if df_sp.empty:
                    print(f"  For xB={xb:g}, no SinglePhase data found.")
                    continue
                    
                df_sp['PhaseAlpha'] = df_sp['PhaseAlpha'].astype(str).str.strip()
                pivot_df = df_sp.pivot_table(index='T', columns='PhaseAlpha', values='G_min', aggfunc='min').sort_index()
                
                # Interpolate missing energies (fixes gaps caused by solver convergence failures)
                pivot_df = pivot_df.interpolate(method='index')

                if 'FCC' not in pivot_df.columns or 'Liquid' not in pivot_df.columns:
                    print(f"  For xB={xb:g}, missing FCC or Liquid SinglePhase data.")
                    continue

                last_fcc_preferred_t = None
                found_transition = False
                for t, row in pivot_df.iterrows():
                    g_fcc = row.get('FCC')
                    g_liq = row.get('Liquid')
                    
                    if pd.isna(g_fcc) or pd.isna(g_liq):
                        continue
                        
                    if g_fcc < g_liq:
                        last_fcc_preferred_t = t
                    elif g_liq < g_fcc and last_fcc_preferred_t is not None:
                        t_melt = (last_fcc_preferred_t + t) / 2.0
                        print(f"  For xB={xb:g}, T_melt = {t_melt:g}  (Averaged T={last_fcc_preferred_t} and T={t})")
                        found_transition = True
                        break
                
                if not found_transition:
                    print(f"  For xB={xb:g}, no FCC->Liquid SinglePhase energy crossing found.")

if __name__ == "__main__":
    find_melting_transitions()