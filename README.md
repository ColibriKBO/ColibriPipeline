# ColibriPipeline
Tools for Colibri data reduction and management

### KernelGeneratorGUI_RAB032322
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
