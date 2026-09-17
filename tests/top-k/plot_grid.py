import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def plot_best_impl(csv_path, device_name, k):
    # 1. Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    # Verify required columns are present
    required_cols = {'DEV', 'K', 'IMPL', 'NCOLS', 'NROWS', 'TIME'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV is missing required columns. Expected: {required_cols}")
        return

    # 2. Filter rows matching the DEV and K given by the user
    df_filtered = df[(df['DEV'] == device_name) & (df['K'] == k)].copy()

    if df_filtered.empty:
        print(f"No data matches DEV='{device_name}' and K={k}.")
        return

    # Establish constant color mapping based on ALL unique IMPL values in the entire CSV
    # This ensures that an IMPL always gets the same color regardless of the DEV/K query
    all_unique_impls = sorted(df['IMPL'].dropna().unique())
    impl_to_id = {impl: i for i, impl in enumerate(all_unique_impls)}
    
    # Create a colormap
    base_cmap = plt.cm.get_cmap('tab20', len(all_unique_impls))
    cmap = mcolors.ListedColormap([base_cmap(i) for i in range(len(all_unique_impls))])
    cmap.set_bad(color='lightgray') # Color for missing NCOLS/NROWS combinations
    
    bounds = np.arange(len(all_unique_impls) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 3. For each pair of NCOLS, NROWS, find IMPL with lowest TIME
    # We group by NCOLS and NROWS, find the index of the min TIME, and extract those rows
    idx_min_time = df_filtered.groupby(['NCOLS', 'NROWS'])['TIME'].idxmin()
    best_impls = df_filtered.loc[idx_min_time].copy()
    
    # Map the winning IMPL to its integer ID for the colormap
    best_impls['IMPL_ID'] = best_impls['IMPL'].map(impl_to_id)

    # 4. Plot the grid
    # Pivot tables to create 2D grids for colors (IDs) and labels (Strings)
    grid_colors = best_impls.pivot(index='NROWS', columns='NCOLS', values='IMPL_ID')
    grid_labels = best_impls.pivot(index='NROWS', columns='NCOLS', values='IMPL')

    # Sort the axes ascending so origin (lowest NCOLS/NROWS) is at bottom-left
    grid_colors = grid_colors.sort_index(ascending=True)
    grid_colors = grid_colors[sorted(grid_colors.columns)]
    grid_labels = grid_labels.sort_index(ascending=True)
    grid_labels = grid_labels[sorted(grid_labels.columns)]

    # Mask missing values so they don't break the integer color mapping
    Z = np.ma.masked_invalid(grid_colors.values)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the grid
    im = ax.imshow(Z, cmap=cmap, norm=norm, origin='lower', aspect='auto')

    # Configure axes ticks
    ax.set_xticks(np.arange(len(grid_colors.columns)))
    ax.set_yticks(np.arange(len(grid_colors.index)))
    ax.set_xticklabels(grid_colors.columns)
    ax.set_yticklabels(grid_colors.index)

    # Grid lines to separate cells
    ax.set_xticks(np.arange(len(grid_colors.columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(grid_colors.index) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate each cell with the string IMPL name
    for i in range(len(grid_labels.index)):
        for j in range(len(grid_labels.columns)):
            impl_name = grid_labels.iat[i, j]
            if pd.notna(impl_name):
                # Using a tiny white bounding box to ensure text is readable over any color
                ax.text(j, i, impl_name, ha="center", va="center", color="black", 
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, edgecolor='none'))

    # Labels and Title
    ax.set_xlabel('NCOLS')
    ax.set_ylabel('NROWS')
    ax.set_title(f'Best IMPL by Execution Time\nDEV: {device_name} | K: {k}', pad=15)

    # Create Legend (Only show IMPLs that actually appeared as a "best" in this view)
    used_impls = sorted(best_impls['IMPL'].unique())
    legend_patches = [
        mpatches.Patch(color=base_cmap(impl_to_id[impl]), label=impl) 
        for impl in used_impls
    ]
    ax.legend(handles=legend_patches, title='Best IMPL', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the fastest implementations for a given Device and K.")
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file")
    parser.add_argument("device_name", type=str, help="Device name to filter by (DEV column)")
    parser.add_argument("k", type=int, help="Integer K to filter by (K column)")
    
    args = parser.parse_args()
    
    plot_best_impl(args.csv_file, args.device_name, args.k)

