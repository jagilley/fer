import os
import glob
from process_pb import parse_args, main

def batch_process_cppnx():
    """
    Batch process all XML files in cppnx/CanalizationPicbreederGenomes
    """
    xml_dir = "cppnx/CanalizationPicbreederGenomes"
    output_base = "data"

    # Get all XML files
    xml_files = glob.glob(f"{xml_dir}/*.xml")
    print(f"Found {len(xml_files)} XML files to process")

    for xml_file in xml_files:
        # Extract the base name (e.g., "576_Skull" from "576_Skull.xml")
        base_name = os.path.basename(xml_file).replace(".xml", "")
        output_dir = os.path.join(output_base, f"picbreeder_{base_name}")

        print(f"\nProcessing: {base_name}")

        # Create args for the main function
        args = parse_args(["--file_path", xml_file, "--save_dir", output_dir])

        try:
            main(args)
            print(f"✓ Saved to {output_dir}")
        except Exception as e:
            print(f"✗ Error processing {base_name}: {e}")
            continue

    print(f"\n\nBatch processing complete! Processed {len(xml_files)} genomes")

if __name__ == "__main__":
    batch_process_cppnx()
