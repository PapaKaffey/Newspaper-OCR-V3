import os
from pathlib import Path
import sys
import datetime

def find_jp2_files(base_dir):
    """Find all JP2 files in the base directory recursively."""
    jp2_files = []
    print(f"Scanning for JP2 files in: {base_dir}")
    
    for root, dirs, files in os.walk(base_dir):
        # Skip debug directories
        if any(pattern in root for pattern in ["_debug", "_processed"]):
            continue
        
        for file in files:
            if file.lower().endswith('.jp2'):
                jp2_files.append(Path(root) / file)
                
    return jp2_files

def check_processing_status(jp2_files, output_dir):
    """Check processing status for each JP2 file."""
    fully_processed = []
    partially_processed = []
    unprocessed = []
    
    for jp2_path in jp2_files:
        base_name = jp2_path.stem
        txt_path = output_dir / f"{base_name}.txt"
        json_path = output_dir / f"{base_name}_meta.json"
        
        # Check if output files exist
        txt_exists = txt_path.exists()
        json_exists = json_path.exists()
        
        if txt_exists and json_exists:
            fully_processed.append(jp2_path)
        elif txt_exists or json_exists:
            partially_processed.append((jp2_path, txt_exists, json_exists))
        else:
            unprocessed.append(jp2_path)
            
    return fully_processed, partially_processed, unprocessed

def main():
    # Configure paths
    input_dir = Path(r"E:\Newspaper OCR V3\downloads\batch_nbu_plattsmouth01_ver01")
    output_dir = Path(r"E:\Newspaper OCR V3\Processed OCR Text")
    
    # Verify directories exist
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return 1
    
    print(f"=== OCR Processing Verification Report ===")
    print(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"----------------------------------------")
    
    # Find all JP2 files
    jp2_files = find_jp2_files(input_dir)
    print(f"Found {len(jp2_files)} JP2 files.")
    
    if not jp2_files:
        print("No JP2 files found. Nothing to verify.")
        return 0
    
    # Check processing status
    print("Checking processing status...")
    fully_processed, partially_processed, unprocessed = check_processing_status(jp2_files, output_dir)
    
    # Calculate percentages
    total = len(jp2_files)
    fully_pct = (len(fully_processed) / total * 100) if total > 0 else 0
    partially_pct = (len(partially_processed) / total * 100) if total > 0 else 0
    unprocessed_pct = (len(unprocessed) / total * 100) if total > 0 else 0
    
    # Generate report
    print("\n=== Processing Status Summary ===")
    print(f"Total JP2 files: {total}")
    print(f"Fully processed files: {len(fully_processed)} ({fully_pct:.1f}%)")
    print(f"Partially processed files: {len(partially_processed)} ({partially_pct:.1f}%)")
    print(f"Unprocessed files: {len(unprocessed)} ({unprocessed_pct:.1f}%)")
    
    # Report partially processed details
    if partially_processed:
        print("\n=== Partially Processed Files ===")
        for jp2_path, has_txt, has_json in partially_processed:
            missing = "TXT" if not has_txt else "JSON"
            print(f"Missing {missing}: {jp2_path}")
    
    # Report unprocessed details (limit to 20 for console output)
    if unprocessed:
        print("\n=== Unprocessed Files ===")
        for i, jp2_path in enumerate(unprocessed):
            if i < 20:  # Show only first 20 for readability
                print(f"{jp2_path}")
            else:
                print(f"... and {len(unprocessed) - 20} more files")
                break
    
    # Generate output report file
    report_path = Path("ocr_verification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"=== OCR Processing Verification Report ===\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Directory: {input_dir}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"----------------------------------------\n\n")
        
        f.write(f"=== Processing Status Summary ===\n")
        f.write(f"Total JP2 files: {total}\n")
        f.write(f"Fully processed files: {len(fully_processed)} ({fully_pct:.1f}%)\n")
        f.write(f"Partially processed files: {len(partially_processed)} ({partially_pct:.1f}%)\n")
        f.write(f"Unprocessed files: {len(unprocessed)} ({unprocessed_pct:.1f}%)\n\n")
        
        if partially_processed:
            f.write("=== Partially Processed Files ===\n")
            for jp2_path, has_txt, has_json in partially_processed:
                missing = "TXT" if not has_txt else "JSON"
                f.write(f"Missing {missing}: {jp2_path}\n")
            f.write("\n")
        
        if unprocessed:
            f.write("=== Unprocessed Files ===\n")
            for jp2_path in unprocessed:
                f.write(f"{jp2_path}\n")
    
    print(f"\nFull report saved to: {report_path}")
    
    # Create files needed to process the unprocessed files
    if unprocessed:
        # Save list of unprocessed files
        unprocessed_list_path = Path("unprocessed_files_list.txt")
        with open(unprocessed_list_path, "w") as f:
            for jp2_path in unprocessed:
                f.write(f"{jp2_path}\n")
        
        print(f"List of unprocessed files saved to: {unprocessed_list_path}")
        
        # Create a batch file that processes the missing files
        batch_path = Path("process_missing_files.bat")
        with open(batch_path, "w") as f:
            f.write("@echo off\n")
            f.write("echo Processing missing JP2 files...\n")
            f.write("echo This will use the default configuration to process all missing files.\n")
            f.write("\n")
            f.write(f'python "E:\\Newspaper OCR V3\\Scripts\\OCR_PIPELINE_VISION_PRIMARY.py" --config "E:\\Newspaper OCR V3\\Scripts\\config_enhanced.yml"\n')
            f.write("\n")
            f.write("echo Done processing missing files.\n")
            f.write("pause\n")
        
        print(f"Batch file to process missing files saved to: {batch_path}")
        print("Run this batch file to process the missing files.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())