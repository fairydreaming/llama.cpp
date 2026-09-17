import argparse
import os

def clean_cuda_logs(input_file, output_file=None):
    # Strings to remove (including their trailing newlines)
    strings_to_remove = [
        "ggml_backend_cuda_graph_compute:",
        "CUDA graph warmup reset\n",
        "CUDA graph warmup complete\n",
        "ggml_cuda_graph_set_enabled: disabling CUDA graphs due to GPU architecture\n"
    ]
    
    # If no output file is specified, we'll write to a temporary file and replace the original
    inplace = False
    if output_file is None:
        output_file = input_file + ".tmp"
        inplace = True

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                # Replace the target strings with an empty string
                for target in strings_to_remove:
                    line = line.replace(target, "")
                
                # If the line contained ONLY the target string, it will now be empty.
                # Writing an empty string does nothing, effectively removing the line.
                outfile.write(line)
                
        # Replace the original file if modifying in-place
        if inplace:
            os.replace(output_file, input_file)
            print(f"Successfully cleaned {input_file}")
        else:
            print(f"Successfully cleaned logs and saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove CUDA graph warmup messages from a log file.")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("-o", "--output_file", help="Path to the output file (optional). If not provided, overwrites the input file.", default=None)
    
    args = parser.parse_args()
    
    clean_cuda_logs(args.input_file, args.output_file)

