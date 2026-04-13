import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# --- USER SETTINGS ---
# Set your target coordinates here
TARGET_T = 1310.0         # Target Temperature
TARGET_XB = 0.99          # Target Composition (xB_total)
TARGET_N = 5e-17          # Target n_total (set to None if you want to see all sizes)

def main():
    # 1. Open file dialog to select the CSV
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Data CSV File to Inspect",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Loading file: {os.path.basename(file_path)}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. Filter the DataFrame
    print(f"\nSearching for point: T = {TARGET_T}, xB_total = {TARGET_XB}" + (f", n_total = {TARGET_N}" if TARGET_N is not None else ""))
    
    # Using np.isclose to safely compare floating point numbers
    mask = np.isclose(df['T'], TARGET_T, atol=1e-3) & np.isclose(df['xB_total'], TARGET_XB, atol=1e-4)
    
    if TARGET_N is not None:
        mask &= np.isclose(df['n_total'], TARGET_N, rtol=1e-5)

    result = df[mask]

    # 3. Display the details
    if result.empty:
        print("\nNo matching points found in the selected file.")
    else:
        print(f"\nFound {len(result)} matching configuration(s):\n")
        for idx, row in result.iterrows():
            print(f"--- Row Index: {idx} ---")
            for col in result.columns:
                print(f"{col:>15}: {row[col]}")
            print("-" * 30)

if __name__ == "__main__":
    main()