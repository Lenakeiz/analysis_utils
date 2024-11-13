import os
import shutil
import matplotlib.pyplot as plt

def savefiguretofile(filename, directory, dpi=300, transparent=False, png_only=True):
    filename_pdf = os.path.join(directory, f"{filename}.pdf")
    filename_png = os.path.join(directory, f"{filename}.png")
    filename_svg = os.path.join(directory, f"{filename}.svg")

    if not png_only:
        plt.savefig(filename_pdf, format='pdf', dpi=dpi, bbox_inches='tight', transparent=transparent)
        plt.savefig(filename_svg, format='svg', dpi=dpi, bbox_inches='tight', transparent=transparent)
        
    plt.savefig(filename_png, format='png', dpi=dpi, bbox_inches='tight', transparent=transparent)

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def move_and_rename_csv_files(origin_folder, destination_folder):
    """
    Move all CSV files from origin_folder to destination_folder, renaming them to remove [1] before .csv if present.
    
    Parameters:
    - origin_folder: Path to the folder containing the CSV files.
    - destination_folder: Path to the folder where files should be copied.
    """
    
    # Check if the origin folder exists
    if not os.path.exists(origin_folder):
        print(f"Error: Origin folder '{origin_folder}' does not exist.")
        return
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Walk through all subdirectories and files in the origin folder
    for root, dirs, files in os.walk(origin_folder):
        for file in files:
            if file.endswith('.csv'):
                # Rename the file by removing '[1]' if it exists and the _en if I apply this after I have decrypted the files
                new_file_name = file.replace('[1]', '')
                new_file_name = new_file_name.replace('_en', '')

                # Full paths for source and destination
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, new_file_name)

                if os.path.abspath(source_path) == os.path.abspath(destination_path):
                    print(f"Skipping file '{source_path}' as it is already in the destination.")
                    continue

                # Copy the file to the new destination
                shutil.copy2(source_path, destination_path)

    print(f"All CSV files have been successfully moved from '{origin_folder}' to '{destination_folder}'.")