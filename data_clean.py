# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 19:49:41 2022

@author: Roman A.
"""
import argparse
import datetime
import sys
from pathlib import Path
import shutil

if len(sys.argv) == 2:
    arg_parser = argparse.ArgumentParser(description=""" Run secondary Colibri processing
        Usage:
    
        """,
        formatter_class=argparse.RawTextHelpFormatter)
    
    arg_parser.add_argument('-d', '--date', help='Observation date (YYYY/MM/DD) of data to be processed.')
    # arg_parser.add_argument('-p', '--procdate', help='Processing date.', default=obs_date)
    cml_args = arg_parser.parse_args()
    obsYYYYMMDD = cml_args.date
    obs_date = datetime.date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))
else:
    obs_date='2022-09-30'

night_dir=str(obs_date)

data_path=Path('/','D:','/ColibriData',str(night_dir).replace('-', ''))

print(data_path)

    #subdirectories of minute-long datasets (~2400 images per directory)
minute_dirs = [f for f in data_path.iterdir() if f.is_dir()]  


#remove bias directory from list of image directories and sort
minute_dirs = [f for f in minute_dirs if 'Bias' not in f.name]

minute_dirs.sort() #list of minute folders

#Create a variable for the file name  
file = data_path.joinpath("to_be_saved.txt")

#Open the file
infile = open(file, 'r') 

lines = infile.readlines() 
saved_times=[]
for line in lines:
	saved_times.append(line.split(' ')[1].replace(':','.')[:-4])  # separates line 
	

infile.close()  


    
for dirs in minute_dirs:
    if dirs.name.split('_')[1] not in saved_times:
        shutil.rmtree(dirs)
        #print(dirs)
        
    