import re
import csv
import argparse
import sys

def extract_metrics_to_csv(input_file, output_file=None):
    # Regex pattern breakdown:
    # ne=\[(\d+),(\d+)         --> Captures NCOLS (Group 1) and NROWS (Group 2)
    # k=(\d+)                  --> Captures K (Group 3)
    # -\s+([0-9.]+)\s+us/run   --> Captures TIME (Group 4)
    pattern = re.compile(r"TOP_K\(.*?ne=\[(\d+),(\d+),.*?\].*?k=(\d+).*?-\s+([0-9.]+)\s+us/run")
    pattern_dev = re.compile(r"Device description: (.*)")
    pattern_impl = re.compile(r"impl_(.*)\.txt")
    
    match = pattern_impl.search(input_file);
    if match:
        impl = match.group(1)

    # Decide where to output: a file or standard output (console)
    out_f = open(output_file, 'w', newline='', encoding='utf-8') if output_file else sys.stdout
    
    try:
        writer = csv.writer(out_f)
        # Write the CSV header
#        writer.writerow(['DEV', 'K', 'IMPL', 'NCOLS', 'NROWS', 'TIME'])
        
        with open(input_file, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                match = pattern_dev.search(line)
                if match:
                    dev = match.group(1)
                    continue
                match = pattern.search(line)
                if match:
                    ncols = int(match.group(1))
                    nrows = int(match.group(2))
                    k = int(match.group(3))
                    time_val = float(match.group(4))
                    
                    # Write the extracted values in the requested order
                    writer.writerow([dev, k, impl, ncols, nrows, time_val])
                    
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    finally:
        # Close the output file only if we opened a real file (not stdout)
        if output_file and out_f:
            out_f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DEV, K, IMPL, NCOLS, NROWS, and TIME from TOP_K log lines to CSV.")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("-o", "--output_file", help="Path to the output CSV file (optional). Prints to console if omitted.", default=None)
    
    args = parser.parse_args()
    
    extract_metrics_to_csv(args.input_file, args.output_file)
