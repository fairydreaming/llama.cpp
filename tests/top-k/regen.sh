#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define base directories
RAW_DIR="logs_raw"
CLEAN_DIR="logs_clean"
CSV_DIR="csv"

mkdir -p "$CSV_DIR"

echo "=== Step 1: Cleaning logs ==="
# Find all .txt files in logs_raw (and its subdirectories)
find "$RAW_DIR" -type f -name "*.txt" -print0 | while IFS= read -r -d '' raw_file; do
    # Remove the starting "logs_raw/" from the path to get the relative structure
    rel_path="${raw_file#"$RAW_DIR/"}"
    out_file="$CLEAN_DIR/$rel_path"
    
    # Create the intermediate directories for the output file
    mkdir -p "$(dirname "$out_file")"
    
    echo "Cleaning: $raw_file -> $out_file"
    python3 clean_logs.py "$raw_file" -o "$out_file"
done

echo "=== Step 2: Converting to CSV ==="
# Create the csv directory if it doesn't exist
mkdir -p "$CSV_DIR"

# Find all immediate subdirectories in logs_clean
find "$CLEAN_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' subdir_path; do
    subdir_name="$(basename "$subdir_path")"
    csv_out="$CSV_DIR/${subdir_name}.csv"
    
    # Empty out the CSV file in case the script is run multiple times
    echo "DEV,K,IMPL,NCOLS,NROWS,TIME" > "$csv_out"
    
    echo "Processing subdirectory: $subdir_name -> $csv_out"
    
    # Find all txt files inside this subdirectory and append their processing output to the csv file
    find "$subdir_path" -type f -name "*.txt" -print0 | while IFS= read -r -d '' txt_file; do
        python3 logs_to_csv.py "$txt_file" >> "$csv_out"
    done
done

echo "=== Step 3: Plotting CSVs ==="
# Find all .csv files in the csv directory
find "$CSV_DIR" -maxdepth 1 -type f -name "*.csv" -print0 | while IFS= read -r -d '' csv_file; do
    # Replace the .csv extension with .png for the output plot name
    plot_file="${csv_file%.csv}.png"
    plot_file=$(basename "$plot_file")
    
    echo "Plotting: $csv_file -> $plot_file"
    python3 plot_nested_grid3.py "$csv_file" -o "$plot_file"
done

echo "All steps completed successfully!"
