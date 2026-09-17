import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import hashlib
import colorsys

def plot_relative_perf_grid(csv_path):
    # 1. Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    required_cols = {'DEV', 'K', 'IMPL', 'NCOLS', 'NROWS', 'TIME'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV is missing required columns. Expected: {required_cols}")
        return

    # Drop any rows with missing essential data and ensure TIME > 0 to avoid division by zero
    df = df.dropna(subset=list(required_cols)).copy()
    df['TIME'] = np.maximum(df['TIME'], 1e-12)

    # Extract unique sorted dimensions
    ncols_vals = sorted(df['NCOLS'].unique())
    nrows_vals = sorted(df['NROWS'].unique())
    k_vals = sorted(df['K'].unique())
    dev_vals = sorted(df['DEV'].unique())

    nc, nr = len(ncols_vals), len(nrows_vals)
    nk, nd = len(k_vals), len(dev_vals)

    # Establish constant color mapping based on ALL unique IMPL values
    all_unique_impls = sorted(df['IMPL'].unique())
    
    def get_constant_color(s):
        # Use MD5 to get a deterministic float [0, 1] for the Hue
        h = int(hashlib.sha1(s.encode('utf-8')).hexdigest()[:8], 16) / 0xffffffff
        return colorsys.hls_to_rgb(h, 0.6, 0.7) # Constant lightness (0.6) and saturation (0.7)

    impl_color_map = {impl: get_constant_color(impl) for impl in all_unique_impls}

    # Group by the dimensional keys to get the minimum time per IMPL (in case of duplicates)
    df_grouped = df.groupby(['NCOLS', 'NROWS', 'DEV', 'K', 'IMPL'])['TIME'].min().reset_index()
    
    # Calculate inverse times and normalize them to create proportional heights
    df_grouped['INV_TIME'] = 1.0 / df_grouped['TIME']
    
    # Calculate the sum of INV_TIME for each specific inner cell
    totals = df_grouped.groupby(['NCOLS', 'NROWS', 'DEV', 'K'])['INV_TIME'].transform('sum')
    
    # The weight is the proportional thickness (height) of the bar
    df_grouped['WEIGHT'] = df_grouped['INV_TIME'] / totals
    
    # Sort values to ensure stacked bars always draw in the same vertical order
    df_grouped = df_grouped.sort_values(['NCOLS', 'NROWS', 'DEV', 'K', 'IMPL'])

    # Mappings for index lookups
    c_idx_map = {val: i for i, val in enumerate(ncols_vals)}
    r_idx_map = {val: i for i, val in enumerate(nrows_vals)}
    k_idx_map = {val: i for i, val in enumerate(k_vals)}
    d_idx_map = {val: i for i, val in enumerate(dev_vals)}

    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Set the background to gray so missing combinations show up naturally
    ax.set_facecolor('lightgray')

    # Prepare patches for the collection
    patches = []
    facecolors = []

    # Map the data into rectangular patches
    # Each cell is a 1x1 block in the coordinate system
    for (ncols, nrows, dev, k), group in df_grouped.groupby(['NCOLS', 'NROWS', 'DEV', 'K']):
        c_idx = c_idx_map[ncols]
        r_idx = r_idx_map[nrows]
        k_idx = k_idx_map[k]
        d_idx = d_idx_map[dev]

        # Calculate base coordinates for this cell
        global_x = c_idx * nk + k_idx
        global_y = r_idx * nd + d_idx
        
        y_offset = global_y
        
        for _, row in group.iterrows():
            w = row['WEIGHT']
            impl = row['IMPL']
            
            # Draw a rectangle: (x, y), width=1, height=weight
            rect = mpatches.Rectangle((global_x, y_offset), 1, w)
            patches.append(rect)
            facecolors.append(impl_color_map[impl])
            
            y_offset += w

    # Add all rectangles to the axes efficiently
    collection = PatchCollection(patches, facecolors=facecolors, edgecolors='none')
    ax.add_collection(collection)

    # Set exact limits of our coordinate system
    ax.set_xlim(0, nc * nk)
    ax.set_ylim(0, nr * nd)

    # Formatting Outer Axes
    # Center the ticks for NCOLS and NROWS in the middle of their blocks
    ax.set_xticks([c * nk + nk / 2 for c in range(nc)])
    ax.set_yticks([r * nd + nd / 2 for r in range(nr)])
    ax.set_xticklabels(ncols_vals)
    ax.set_yticklabels(nrows_vals)
    ax.set_xlabel('NCOLS (Outer X)')
    ax.set_ylabel('NROWS (Outer Y)')
    ax.set_title('Relative Performance by IMPL (Inverse Execution Time)', pad=15)

    # Draw THICK major gridlines to separate the Outer blocks (NCOLS / NROWS)
    for c in range(1, nc):
        ax.axvline(c * nk, color='black', linewidth=2)
    for r in range(1, nr):
        ax.axhline(r * nd, color='black', linewidth=2)
        
    # Draw minor gridlines to separate Inner blocks faintly
    for x in range(1, nc * nk):
        if x % nk != 0:
            ax.axvline(x, color='white', linewidth=0.5, alpha=0.5)
    for y in range(1, nr * nd):
        if y % nd != 0:
            ax.axhline(y, color='white', linewidth=0.5, alpha=0.5)

    # Disable tick marks visually but keep labels
    ax.tick_params(which="both", bottom=False, left=False)

    # Create Color Legend
    legend_patches = [
        mpatches.Patch(color=impl_color_map[impl], label=impl) 
        for impl in all_unique_impls
    ]
    
    # Legend 1: Colors
    leg_colors = ax.legend(handles=legend_patches, title='Implementation', 
                           bbox_to_anchor=(1.03, 1), loc='upper left')
    ax.add_artist(leg_colors)

    # Legend 2: Explanation of the subgrid layout
    k_str = ", ".join(map(str, k_vals))
    dev_str = "\n".join(map(str, dev_vals))
    
    subgrid_expl = (
        "Subgrid Layout (Inside each cell):\n\n"
        "X-axis ➔ K values:\n"
        f"[{k_str}]\n\n"
        "Y-axis ➔ DEV values (Bottom to Top):\n"
        f"{dev_str}\n\n"
        "Bar Height ➔ Proportional to\n"
        "1 / TIME (Thicker = Faster)"
    )
    
    # Add textual explanation box below the main legend
    ax.text(1.03, 0.5, subgrid_expl, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', fc='#f9f9f9', ec='gray', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(right=0.75) # make room for text on the right
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a nested grid of relative implementation performances.")
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file")
    
    args = parser.parse_args()
    plot_relative_perf_grid(args.csv_file)
