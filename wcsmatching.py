"""
Filename:   wcsmatching.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    June 19, 2022
Updated:    June 20, 2022
    
Usage: python wcsmatching.py <obs_date>
"""

# Module Imports
import os,sys
import argparse
import pathlib
import multiprocessing
import gc
import sep
import re
import numpy as np
import time as timer
from datetime import datetime,timedelta
from astropy.convolution import RickerWavelet1DKernel
from astropy.time import Time
from copy import deepcopy
from multiprocessing import Pool

# Custom Script Imports
import getRAdec
import colibri_image_reader as cir
import colibri_photometry as cp
from coordsfinder import getTransform

# Disable Warnings
import warnings
import logging
#warnings.filterwarnings("ignore",category=DeprecationWarning)
#warnings.filterwarnings("ignore",category=VisibleDeprecationWarning)

#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = pathlib.Path('D:/')
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'
