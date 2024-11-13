import os
import matplotlib.pyplot as plt

def savefiguretofile(filename, directory, dpi=300, transparent=False, png_only=True):
    filename_pdf = os.path.join(directory, f"{filename}.pdf")
    filename_png = os.path.join(directory, f"{filename}.png")
    filename_svg = os.path.join(directory, f"{filename}.svg")

    if not png_only:
        plt.savefig(filename_pdf, format='pdf', dpi=dpi, bbox_inches='tight', transparent=transparent)
        plt.savefig(filename_svg, format='svg', dpi=dpi, bbox_inches='tight', transparent=transparent)
        
    plt.savefig(filename_png, format='png', dpi=dpi, bbox_inches='tight', transparent=transparent)