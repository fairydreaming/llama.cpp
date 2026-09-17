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

# Predefined base colors for known implementations
IMPL_PALETTE = {
    'bitonic': '#1f77b4',           # Blue
    'argsort': '#000000',           # Black
    'radix': '#2ca02c',             # Green
    'DeviceTopK': '#d62728',        # Red
    'DeviceBatchedTopK': '#9467bd'  # Purple
}

def get_gpu_release_date(gpu_name: str) -> float:
    """
    Returns an approximate release date (Year.Month) for a given NVIDIA GPU.
    Used as a sorting key.
    """
    name_upper = gpu_name.upper()

    architecture_patterns = {
        r"RTX\s*50\d\d": 2025.01,
        r"\bB100\b|\bB200\b": 2024.03,
        r"RTX\s*40\d\d": 2022.09,
        r"\bH100\b|\bH200\b|\bL40\b": 2022.03,
        r"RTX\s*30\d\d": 2020.09,
        r"\bA100\b|\bA10\b|\bA30\b|\bA40\b|\bA6000\b": 2020.05,
        r"RTX\s*20\d\d": 2018.09,
        r"GTX\s*16\d\d": 2019.02,
        r"\bT4\b": 2018.09,
        r"\bV100\b|\bTITAN\s*V\b": 2017.06,
        r"GTX\s*10\d\d": 2016.05,
        r"\bP100\b": 2016.04,
        r"\bP40\b": 2016.09,
        r"\bP4\b": 2016.09,
        r"GTX\s*9\d\d": 2014.09,
        r"\bM40\b|\bM60\b": 2015.11,
        r"GTX\s*7\d\d": 2013.05,
        r"\bK80\b|\bK40\b": 2014.11,
    }

    for pattern, release_date in architecture_patterns.items():
        if re.search(pattern, name_upper):
            return release_date

    return 9999.99

def sort_nvidia_gpus(gpu_list: List[str]) -> List[str]:
    """
    Sorts a list of NVIDIA GPUs in historical order from oldest to newest.
    """
    return sorted(gpu_list, key=get_gpu_release_date)

def get_base_color(impl: str) -> tuple:
    """
    Returns the RGB base color for an implementation.
    Uses predefined palette if available, otherwise generates a deterministic color.
    """
    if impl in IMPL_PALETTE:
        return mcolors.to_rgb(IMPL_PALETTE[impl])
    
    # Fallback: Hash-based deterministic color
    h = int(hashlib.sha1(impl.encode('utf-8')).hexdigest()[:8], 16) / 0xffffffff
    return colorsys.hls_to_rgb(h, 0.6, 0.7)

def plot_nested_grid(csv_path, output):
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

    all_unique_impls = sorted(df['IMPL'].dropna().unique())
    
    # Sort dataframe by dimensions and then by time to easily grab top 2
    df_sorted = df.sort_values(by=['NCOLS', 'NROWS', 'DEV', 'K', 'TIME']).dropna()
    grouped = df_sorted.groupby(['NCOLS', 'NROWS', 'DEV', 'K'])

    # Initialize a large 3D array for the composite RGB grid (filled with white/NaN equivalent)
    # Background will be white (1.0, 1.0, 1.0)
    Z_rgb = np.ones((nr * nd, nc * nk, 3), dtype=np.float32)

    # Mappings for index lookups
    c_idx_map = {val: i for i, val in enumerate(ncols_vals)}
    r_idx_map = {val: i for i, val in enumerate(nrows_vals)}
    k_idx_map = {val: i for i, val in enumerate(k_vals)}
    d_idx_map = {val: i for i, val in enumerate(dev_vals)}

    # Process each configuration group
    for name, group in grouped:
        c_val, r_val, d_val, k_val = name
        
        times = group['TIME'].values
        impls = group['IMPL'].values
        
        best_impl = impls[0]
        t_fastest = times[0]
        
        # Calculate speedup relative to the second fastest
        if len(times) > 1 and t_fastest > 0:
            t_second = times[1]
            speedup = t_second / t_fastest
        else:
            speedup = 1.0 # Only one implementation available, no relative speedup
            
        # Get coordinates
        c_idx = c_idx_map[c_val]
        r_idx = r_idx_map[r_val]
        k_idx = k_idx_map[k_val]
        d_idx = d_idx_map[d_val]

        global_x = c_idx * nk + k_idx
        global_y = r_idx * nd + d_idx
        
        # Modulate Brightness based on speedup
        # speedup usually > 1.0. We cap at 2.0.
        s_capped = min(speedup, 2.0)
        # alpha maps [1.0, 2.0] -> [0.0, 1.0]
        alpha = s_capped - 1.0
        
        # We blend between white (low speedup, very bright/pale) and the base color (high speedup, true color)
        base_r, base_g, base_b = get_base_color(best_impl)
        
        # Pale/white blend: result = base * alpha + white * (1 - alpha)
        # Minimum alpha of 0.15 so it doesn't completely disappear into the background
        alpha_adj = 0.45 + 0.55 * alpha 
        
        r = base_r * alpha_adj + 1.0 * (1 - alpha_adj)
        g = base_g * alpha_adj + 1.0 * (1 - alpha_adj)
        b = base_b * alpha_adj + 1.0 * (1 - alpha_adj)

        Z_rgb[global_y, global_x] = [r, g, b]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(13, 9))
    im = ax.imshow(Z_rgb, origin='lower', aspect='auto')

    # Formatting Outer Axes
    ax.set_xticks([c * nk + (nk - 1) / 2 for c in range(nc)])
    ax.set_yticks([r * nd + (nd - 1) / 2 for r in range(nr)])
    ax.set_xticklabels(ncols_vals)
    ax.set_yticklabels(nrows_vals)
    ax.set_xlabel('NCOLS (Outer X)')
    ax.set_ylabel('NROWS (Outer Y)')
    ax.set_title('Best TOP_K Implementation & Relative Speedup (Nested Grid)', pad=15)

    # Gridlines
    major_xticks = [c * nk - 0.5 for c in range(1, nc)]
    major_yticks = [r * nd - 0.5 for r in range(1, nr)]
    minor_xticks = [x - 0.5 for x in range(1, nc * nk) if x not in [c * nk for c in range(1, nc)]]
    minor_yticks = [y - 0.5 for y in range(1, nr * nd) if y not in [r * nd for r in range(1, nr)]]

    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)

    for x in major_xticks:
        ax.axvline(x, color='black', linewidth=2)
    for y in major_yticks:
        ax.axhline(y, color='black', linewidth=2)

    ax.tick_params(which="both", bottom=False, left=False)

    # Create Color Legend for Implementations
    legend_patches = [
        mpatches.Patch(color=get_base_color(impl), label=impl) 
        for impl in all_unique_impls
    ]
    
    leg_colors = ax.legend(handles=legend_patches, title='Best IMPL (Base Color)', 
                           bbox_to_anchor=(1.03, 1), loc='upper left')
    ax.add_artist(leg_colors)

    # Add Speedup/Brightness Legend Text
    speedup_expl = (
        "Color Brightness (Speedup):\n\n"
        "Pale/White ➔ Narrow Win\n"
        "(Speedup ~1.0x)\n\n"
        "Bold/Pure ➔ Decisive Win\n"
        "(Speedup >= 2.0x)"
    )
    
    ax.text(1.03, 0.65, speedup_expl, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', fc='#f9f9f9', ec='gray', alpha=0.9))

    # Add Subgrid Layout Explanation
    k_str = ", ".join(map(str, k_vals))
    dev_str = "\n".join(map(str, dev_vals))
    
    subgrid_expl = (
        "Subgrid Layout (Inside each cell):\n\n"
        "X-axis ➔ K values:\n"
        f"[{k_str}]\n\n"
        "Y-axis ➔ DEV values (Bottom to Top):\n"
        f"{dev_str}"
    )
    
    ax.text(1.03, 0.40, subgrid_expl, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', fc='#f9f9f9', ec='gray', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap → {output}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a nested grid of fastest implementations with speedup.")
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file")
    parser.add_argument("-o", "--output", default=None, help="Output image file (PNG, PDF, etc.)")
    
    args = parser.parse_args()
    plot_nested_grid(args.csv_file, args.output)
