import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(page_title="llama.cpp TOP_K Benchmark Explorer", layout="wide")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Ensure proper data types
    df['K'] = df['K'].astype(int)
    df['NCOLS'] = df['NCOLS'].astype(int)
    df['NROWS'] = df['NROWS'].astype(int)
    df['TIME'] = df['TIME'].astype(float)
    return df

def main():
    st.title("🦙 llama.cpp TOP_K Kernel Performance Explorer")
    st.markdown("Analyze CUDA kernel performance for different tensor shapes, $K$ values, and implementations.")

    # Sidebar: Data Loading
    st.sidebar.header("1. Load Data")
    uploaded_file = st.sidebar.file_uploader("Upload Benchmark CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        st.info("Please upload your benchmark CSV file in the sidebar to begin. Using sample data for demonstration.")
        # Minimal sample data matching user structure
        sample_csv = """DEV,K,IMPL,NCOLS,NROWS,TIME
Tesla P40,20,bitonic,128,4096,201.47
Tesla P40,20,radix,128,4096,150.00
Tesla P40,20,bitonic,256,4096,570.07
Tesla P40,20,radix,256,4096,600.10
Tesla P40,20,bitonic,1024,2048,2133.59
Tesla P40,20,radix,1024,2048,2500.00
Tesla P40,40,bitonic,128,1,16.17
Tesla P40,40,radix,128,1,12.00
Tesla P40,40,bitonic,256,1,20.09
Tesla P40,40,radix,256,1,22.50
"""
        df = load_data(io.StringIO(sample_csv))

    # Sidebar: Filtering
    st.sidebar.header("2. Filter Parameters")
    
    devices = df['DEV'].unique()
    selected_dev = st.sidebar.selectbox("Select GPU Device (DEV)", devices)
    
    k_values = sorted(df['K'].unique())
    selected_k = st.sidebar.selectbox("Select K value", k_values)

    # Filter data based on selections
    df_filtered = df[(df['DEV'] == selected_dev) & (df['K'] == selected_k)]
    
    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
        return

    st.header(f"Performance for `DEV`: {selected_dev} | `K`: {selected_k}")

    # Data Processing: Find Best and 2nd Best Implementations
    # Sort by shape and time (ascending) to rank implementations
    sorted_df = df_filtered.sort_values(by=['NCOLS', 'NROWS', 'TIME'])
    
    # Get the best (fastest) implementation for each shape
    best_df = sorted_df.groupby(['NCOLS', 'NROWS']).first().reset_index()
    best_df.rename(columns={'IMPL': 'Best_IMPL', 'TIME': 'Best_TIME'}, inplace=True)

    # Get the second best implementation to calculate speedup (if available)
    second_best_df = sorted_df.groupby(['NCOLS', 'NROWS']).nth(1).reset_index()
    second_best_df.rename(columns={'TIME': 'Second_Best_TIME'}, inplace=True)

    # Merge to compute relative speedup
    if not second_best_df.empty:
        win_matrix = pd.merge(best_df, second_best_df[['NCOLS', 'NROWS', 'Second_Best_TIME']], on=['NCOLS', 'NROWS'], how='left')
        win_matrix['Speedup'] = win_matrix['Second_Best_TIME'] / win_matrix['Best_TIME']
        # Fill missing speedup with 1.0 (if only one implementation was tested)
        win_matrix['Speedup'] = win_matrix['Speedup'].fillna(1.0)
    else:
        win_matrix = best_df.copy()
        win_matrix['Speedup'] = 1.0

    # ---------------------------------------------------------
    # CHART 1: Winner Matrix Overview
    # ---------------------------------------------------------
    st.subheader("🏆 Winning Implementation Matrix")
    st.markdown("This chart shows the fastest implementation for each tensor shape. Larger circles mean a greater speedup compared to the 2nd best implementation. **Both axes are log2 scaled.**")

    # Limit marker size for visual clarity
    max_size = 30
    min_size = 10
    
    fig_matrix = px.scatter(
        win_matrix, 
        x="NCOLS", 
        y="NROWS", 
        color="Best_IMPL",
        size="Speedup",
        hover_data=["Best_TIME", "Speedup"],
        log_x=True, 
        log_y=True,
        title=f"Fastest Implementations (K={selected_k})",
        labels={"NCOLS": "Number of Columns (NCOLS)", "NROWS": "Number of Rows (NROWS)", "Best_IMPL": "Winner"}
    )
    
    # Improve axis tick formatting for base 2
    fig_matrix.update_layout(
        xaxis=dict(tickmode='array', tickvals=sorted(df['NCOLS'].unique())),
        yaxis=dict(tickmode='array', tickvals=sorted(df['NROWS'].unique())),
        height=600
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

    # ---------------------------------------------------------
    # CHART 2: Detailed Log-Log Performance Curves
    # ---------------------------------------------------------
    st.subheader("📈 Detailed Performance Curves")
    st.markdown("Select a specific `NROWS` to see how execution time scales as `NCOLS` increases for all implementations.")
    
    available_nrows = sorted(df_filtered['NROWS'].unique())
    selected_nrows = st.selectbox("Select NROWS for line chart detail:", available_nrows)

    curve_data = df_filtered[df_filtered['NROWS'] == selected_nrows].sort_values(by="NCOLS")
    
    fig_curve = px.line(
        curve_data, 
        x="NCOLS", 
        y="TIME", 
        color="IMPL", 
        markers=True,
        log_x=True, 
        log_y=True,
        title=f"Execution Time vs Columns (NROWS = {selected_nrows}, K = {selected_k})",
        labels={"TIME": "Time (µs)", "NCOLS": "Number of Columns (NCOLS)", "IMPL": "Implementation"}
    )
    
    fig_curve.update_layout(
        xaxis=dict(tickmode='array', tickvals=sorted(curve_data['NCOLS'].unique())),
        height=500
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    # ---------------------------------------------------------
    # Raw Data Table
    # ---------------------------------------------------------
    with st.expander("Show Raw Filtered Data"):
        st.dataframe(df_filtered.sort_values(by=['NCOLS', 'NROWS', 'TIME']))

if __name__ == "__main__":
    main()
