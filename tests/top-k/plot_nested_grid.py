import re
import argparse
import pandas as pd
import numpy as np
import hashlib
import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from typing import List

def get_gpu_release_date(gpu_name: str) -> float:
    """
    Returns an approximate release date (Year.Month) for a given NVIDIA GPU.
    Used as a sorting key.
    """
    name_upper = gpu_name.upper()

    # Mapping of regex patterns to approximate release dates (YYYY.MM)
    # Ordered roughly by architecture: Kepler -> Maxwell -> Pascal -> Volta -> Turing -> Ampere -> Hopper/Ada -> Blackwell
    architecture_patterns = {
        # --- Blackwell Architecture (~2024-2025) ---
        r"RTX\s*50\d\d": 2025.01,
        r"\bB100\b|\bB200\b": 2024.03,

        # --- Hopper & Ada Lovelace Architectures (~2022) ---
        r"RTX\s*40\d\d": 2022.09,
        r"\bH100\b|\bH200\b|\bL40\b": 2022.03,

        # --- Ampere Architecture (~2020) ---
        r"RTX\s*30\d\d": 2020.09,
        r"\bA100\b|\bA10\b|\bA30\b|\bA40\b|\bA6000\b": 2020.05,

        # --- Turing Architecture (~2018) ---
        r"RTX\s*20\d\d": 2018.09,
        r"GTX\s*16\d\d": 2019.02, # Budget Turing cards came a bit later
        r"\bT4\b": 2018.09,

        # --- Volta Architecture (~2017) ---
        r"\bV100\b|\bTITAN\s*V\b": 2017.06,

        # --- Pascal Architecture (~2016) ---
        r"GTX\s*10\d\d": 2016.05,
        r"\bP100\b": 2016.04,
        r"\bP40\b": 2016.09,
        r"\bP4\b": 2016.09,

        # --- Maxwell Architecture (~2014-2015) ---
        r"GTX\s*9\d\d": 2014.09,
        r"\bM40\b|\bM60\b": 2015.11,

        # --- Kepler Architecture (~2012-2014) ---
        r"GTX\s*7\d\d": 2013.05,
        r"\bK80\b|\bK40\b": 2014.11,
    }

    # Search the GPU name against our patterns
    for pattern, release_date in architecture_patterns.items():
        if re.search(pattern, name_upper):
            return release_date

    # Fallback for completely unrecognized GPUs: push them to the end of the list
    return 9999.99

def sort_nvidia_gpus(gpu_list: List[str]) -> List[str]:
    """
    Sorts a list of NVIDIA GPUs in historical order from oldest to newest.
    """
    return sorted(gpu_list, key=get_gpu_release_date)

def plot_nested_grid(csv_path, output):
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

    # Extract unique sorted dimensions
    ncols_vals = sorted(df['NCOLS'].dropna().unique())
    nrows_vals = sorted(df['NROWS'].dropna().unique())
    k_vals = sorted(df['K'].dropna().unique())
    dev_vals = sort_nvidia_gpus(df['DEV'].dropna().unique())

    nc, nr = len(ncols_vals), len(nrows_vals)
    nk, nd = len(k_vals), len(dev_vals)

    # Establish constant color mapping based on ALL unique IMPL values
    all_unique_impls = sorted(df['IMPL'].dropna().unique())
    impl_to_id = {impl: i for i, impl in enumerate(all_unique_impls)}
    
#    base_cmap = plt.cm.get_cmap('tab20', len(all_unique_impls))
#    cmap = mcolors.ListedColormap([base_cmap(i) for i in range(len(all_unique_impls))])
    # Generate a deterministic color for each IMPL using a string hash
    def get_constant_color(s):
        # Use MD5 to get a deterministic float [0, 1] for the Hue
        h = int(hashlib.sha1(s.encode('utf-8')).hexdigest()[:8], 16) / 0xffffffff
        return colorsys.hls_to_rgb(h, 0.6, 0.7) # Constant lightness (0.6) and saturation (0.7)

    impl_colors = [get_constant_color(impl) for impl in all_unique_impls]
    cmap = mcolors.ListedColormap(impl_colors)
    cmap.set_bad(color='darkgray') # Color for missing combinations
    
    bounds = np.arange(len(all_unique_impls) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Find IMPL with lowest TIME for each (NCOLS, NROWS, DEV, K) combination
    idx_min_time = df.groupby(['NCOLS', 'NROWS', 'DEV', 'K'])['TIME'].idxmin()
    best_impls = df.loc[idx_min_time].copy()
    best_impls['IMPL_ID'] = best_impls['IMPL'].map(impl_to_id)

    # Initialize a large 2D array for the composite grid (filled with NaNs)
    # Total width = outer NCOLS * inner K
    # Total height = outer NROWS * inner DEV
    Z = np.full((nr * nd, nc * nk), np.nan)

    # Mappings for index lookups
    c_idx_map = {val: i for i, val in enumerate(ncols_vals)}
    r_idx_map = {val: i for i, val in enumerate(nrows_vals)}
    k_idx_map = {val: i for i, val in enumerate(k_vals)}
    d_idx_map = {val: i for i, val in enumerate(dev_vals)}

    # Map the data into the large 2D matrix
    for _, row in best_impls.iterrows():
        c_idx = c_idx_map[row['NCOLS']]
        r_idx = r_idx_map[row['NROWS']]
        k_idx = k_idx_map[row['K']]
        d_idx = d_idx_map[row['DEV']]

        # Calculate coordinates in the flattened image
        global_x = c_idx * nk + k_idx
        global_y = r_idx * nd + d_idx
        
        Z[global_y, global_x] = row['IMPL_ID']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 9))
    Z_masked = np.ma.masked_invalid(Z)
    im = ax.imshow(Z_masked, cmap=cmap, norm=norm, origin='lower', aspect='auto')

    # Formatting Outer Axes
    # Center the ticks for NCOLS and NROWS in the middle of their blocks
    ax.set_xticks([c * nk + (nk - 1) / 2 for c in range(nc)])
    ax.set_yticks([r * nd + (nd - 1) / 2 for r in range(nr)])
    ax.set_xticklabels(ncols_vals)
    ax.set_yticklabels(nrows_vals)
    ax.set_xlabel('NCOLS (Outer X)')
    ax.set_ylabel('NROWS (Outer Y)')
    ax.set_title('Best TOP_K implementation - lowest execution time per run (Nested Grid)', pad=15)

    # Draw THICK major gridlines to separate the Outer blocks (NCOLS / NROWS)
    major_xticks = [c * nk - 0.5 for c in range(1, nc)]
    major_yticks = [r * nd - 0.5 for r in range(1, nr)]
    
    # Draw minor gridlines to separate Inner blocks faintly
    minor_xticks = [x - 0.5 for x in range(1, nc * nk) if x not in [c * nk for c in range(1, nc)]]
    minor_yticks = [y - 0.5 for y in range(1, nr * nd) if y not in [r * nd for r in range(1, nr)]]

    # Apply standard lines for inner boundaries
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)

    # Overlay thick lines manually for outer boundaries so they stand out perfectly
    for x in major_xticks:
        ax.axvline(x, color='black', linewidth=2)
    for y in major_yticks:
        ax.axhline(y, color='black', linewidth=2)

    # Disable tick marks visually but keep labels
    ax.tick_params(which="both", bottom=False, left=False)

    # Create Color Legend
    legend_patches = [
#        mpatches.Patch(color=base_cmap(impl_to_id[impl]), label=impl) 
        mpatches.Patch(color=impl_colors[impl_to_id[impl]], label=impl) 
        for impl in all_unique_impls
    ]
    
    # Legend 1: Colors
    leg_colors = ax.legend(handles=legend_patches, title='Best IMPL', 
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
        f"{dev_str}"
    )
    
    # Add textual explanation box below the main legend
    ax.text(1.03, 0.5, subgrid_expl, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', fc='#f9f9f9', ec='gray', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(right=0.75) # make room for text on the right

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap → {output}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a nested grid of fastest implementations.")
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file")
    parser.add_argument("-o", "--output", default=None, help="Output image file (PNG, PDF, etc.)")
    
    args = parser.parse_args()
    plot_nested_grid(args.csv_file, args.output)
