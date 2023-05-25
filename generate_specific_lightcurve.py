"""
Filename:   generate_specific_lightcurve.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Thu May 25, 2022
Updated:    Thu May 25, 2022
    
Usage: python generate_specific_lightcurve.py <obs_date> <time_of_interest> [RA,DEC]
"""

# Module Imports
import os,sys
import argparse
import pathlib
import multiprocessing
import datetime
import gc
import sep
import numpy as np
import time as timer
from astropy.convolution import RickerWavelet1DKernel
from astropy.time import Time
from copy import deepcopy
from multiprocessing import Pool

# Custom Script Imports
import colibri_image_reader as cir
import colibri_photometry as cp

# Disable Warnings
import warnings
import logging
#warnings.filterwarnings("ignore",category=DeprecationWarning)
#warnings.filterwarnings("ignore",category=VisibleDeprecationWarning)


#--------------------------------functions------------------------------------#

