# ColibriPipeline
Tools for Colibri data reduction and management

### Init/
Directory containing the bash and batch files used to automate the compiling of all cython modules.

### PlotsAndPDFs/
Directory containing any and all figures and PDFs to be kept in this project.
These include, but are not limited to: pipeline flowcharts, usage manuals, and boilerplate Colibri presentations.

### KernelGeneratorGUI_RAB032322/
Copied from Rishi's KernelGeneratorGUI repository. Generates a set of occultation kernels to compare data to. 
Includes changes to fresnelModeler.py which saves the parameters of each kernel. Also includes a script to generate plots of all kerenls generated. 
This is necessary to run ColibriSecondary.py.

### VizieR_query.py
Written by Jilly Ryan, modified by Rachel Brown to make into a single function that can be called by other scripts.
Performs query to Gaia catalog given a set of coordinates and search radius.

### astrometrynet_funcs.py
Upload image to astrometry.net API to get plate solution

### colibri_main_py3.py
Primary pipeline for processing Colibri Data. Identifies candidate events.

### colibri_secondary.py
Secondary pipeline for processing Colibri Data. Matches candidate events with occultation kernels, and gets star info from Gaia (RA/dec, mag)

### getRADec.py
Transforms X,Y coordinates in an image to RA, dec using a WCS transformation from astrometry.net

### lightcurve_looker.py
Creates plots of star light curves.

### lightcurve_maker.py
Modified from colibri_main.py. Finds stars in an image set, does photometry on stars and returns their light curves over the minute.

### sensitivity.py
For estimating the sensitivity of our instruments.
Gets the median flux / stddev for all stars in a minute. Matches the stars to the Gaia catalog and compares SNR vs Gaia magnitude. 

### timingplots.py
Checks files for timestamp issues and corrupted files.

### biasPlots.py - script to quickly see bias and sensor temperature values for specific night, runs on Green ONLY
	I think it can be deleted

### coordsfinder.py - writes Ra Dec coordinates into dip detection txt file by performing AstrometryNet transformation
	Is part of pipeline, runs after colibri_main_pipeline
	
### cumulative_stats.py - Cumulative statistics for detections and occultations, runs only on Green

### data_clean.py - Cleans observed minute dirs in specific night that are not in the list called to_be_saved.txt
	Not implemented yet
	
### getStarHour.py - Return star-hours for each field observed in the night directory based 
	on number of frames of certain fields and number of stars in the mid-frame
	This script is used by timeline.py
	Must be updated if our fields change
	
### readnoise.py - just delete it. It's completely implemented and documented in image_stats_bias.py

### simultaneous_occults.py - Match dip detection txts through-out 3 telescopes based on time and coordinates
	Runs only on Green. Is part of pipeline. I hope one day it will be renamed to colibri_secondary.py

### timeline.py - Create a bunch of observatory operation plots, more description in separate file,
	Runs only on Green
