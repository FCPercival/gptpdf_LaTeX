# Working principle:
# 1. Ask the user for the path to the PDF file (Also by drag and drop into the terminal)
# 2. Ask the user for the path to the output directory (By pre-defined paths)
# 3. Use a boolean variable to select between the original parser or the YOLO image parser
# 4. Parse the PDF file and extract the text and images
# 5. Save the text to a .tex file and images to the output directory

# Before executing the script, make sure to install the required packages:
# pip install PyMuPDF shapely GeneralAgent
# For YOLO detector, additional dependencies are required (see doclayout_yolo.py)
# Also change the path to the gptpdf_LaTeX package in sys.path.append() to your own path,
# the APIKEY in the run_script() function and the paths in the list_of_outputs.

# This example demonstrates two different methods for parsing PDF files:
# 1. YOLO Parser: Uses YOLOv10 to detect figures, then uses LLM only for text extraction
#    - Pros: More accurate figure detection, better handling of complex layouts (better in combination
#            with a CUDA enabled GPU, otherwise gpu inference is available (near one second for page)
#    - Cons: Requires additional dependencies (YOLOv10), a little bit slower
# 2. Original Parser: Uses LLM to detect and extract both text and images
#    - Pros: Works without additional dependencies, simpler setup
#    - Cons: May not detect all figures accurately, especially complex ones

import os
import sys
import dotenv
from gptpdf_LaTeX import parse_pdf # pip install gptpdf_LaTeX

document_initial_text= """
\\documentclass[a4paper,14pt]{extarticle}
\\usepackage{graphicx, mathptmx, amsmath, amsfonts, url, hyperref, tikz, float}
\\usepackage{amsfonts}
\\usepackage{amsmath}
\\usepackage{textcomp}
\\usepackage{xcolor}
\\usepackage{geometry}

\\title{TITLE} %IS or SEF or SPM
\\author{AUTHOR}
\\date{DATE}

\\begin{document}

    \\maketitle
"""

document_final_text= """
\\end{document}
"""

pdf_path = ""
output_dir = ""
output_dir_images = ""

def run_script():
    from gptpdf import parse_pdf
    api_key = "APIKEY" # <----------------------Add your key----------------------
    base_url = "https://api.openai.com/v1"

    # ---- PATH MANAGEMENT ----
    pdf_path = input("Enter the path to the PDF file: ")
    pdf_path = pdf_path.replace('\"', '')
    list_of_outputs = {"KEYTOSHOW": "OUTPUTPATH1", "KEY2": "PATH2"}
    keys = list(list_of_outputs.keys())
    print("Please choose one of the following options:")
    for i, key in enumerate(keys, start=1):
        print(f"{i}. {key}")
    try:
        choice = int(input("Enter the number of your choice: "))
        if 1 <= choice <= len(keys):
            selected_key = keys[choice - 1]
            selected_path = list_of_outputs[selected_key]
            print(f"You selected '{selected_key}' with path: {selected_path}")
            output_dir = selected_path
            output_dir_images = output_dir + '\\images\\'
        else:
            print("Invalid choice. Please run the program again and choose a valid number.")
    except ValueError:
        print("Please enter a valid number.")
        exit()

    # ---- PARSER SELECTION ----
    # Set a boolean variable to select the parser
    # True = YOLO Parser (RECOMMENDED: YOLOv10 detects figures, LLM extracts text only)
    # False = Original Parser (LLM detects and extracts both text and images)
    use_yolo = True

    # Call parse_pdf with the selected parser
    print(f"\nProcessing PDF with {'YOLO' if use_yolo else 'Original'} Parser...")
    content, image_paths = parse_pdf(
        pdf_path, 
        output_dir = output_dir,
        api_key = api_key,
        model = 'gpt-4o',
        gpt_worker = 6,
        document_initial_text = document_initial_text,
        document_final_text = document_final_text,
        base_url = base_url,
        output_dir_images = output_dir_images,
        use_yolo_detector = use_yolo
    )

    print("\nProcessing complete!")
    print(f"Number of images extracted: {len(image_paths)}")
    print("Output LaTeX file generated at:", os.path.join(output_dir, 'output.tex'))

run_script()
exit()
