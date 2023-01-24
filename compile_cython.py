"""
Filename:   compile_cython.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Fri Oct 21 09:23:04 2022
Updated:    Fri Oct 21 09:23:04 2022

Usage: python compile_cython.py
Run this file to compile the cython modules in the working directory.
"""

import os,sys
import platform


##############################
## Compile Cython Modules
##############################

print("Compiling cython files...\n")
for file in os.listdir("."):
    # Detect cython files and compile
    if file.endswith(".pyx"):
        print(f"Compiling {file}")
        if os.path.exists(f"setup_{file[:-4]}.py"):
            os.system(f"python setup_{file[:-4]}.py build_ext --inplace")
        else:
            print(f"No setup file exists for {file}!\n")
            continue

        # Detect platform and clean up compilation
        if platform.system() == "Windows":
            os.system(f"rd /s /q build/")
            os.system(f"del {file[:-4]}.html")
            os.system(f"del {file[:-4]}.c")
            os.system(f"del {file[:-4]}.pyd")
            os.system(f"RENAME {file[:-4]}.*.pyd {file[:-4]}.pyd")

        elif platform.system() == "Linux":
            os.system(f"rm -r build/")
            os.system(f"rm {file[:-4]}.html")
            os.system(f"rm {file[:-4]}.c")
            os.system(f"rm {file[:-4]}.so")
            os.system(f"mv {file[:-4]}.*.so {file[:-4]}.so")


        print("\n")
