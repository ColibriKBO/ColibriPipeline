import os
import subprocess

basedir = 'D:'
path_to_file = '/tmp/airmasssensitivity'
date = '2024/02/21'
path_to_data = basedir + path_to_file + '/' + date.replace('/','')

for filename in os.listdir(path_to_data):

    minute = filename.split('_')[1] 
    
    subprocess.run(['python', 'sensitivity2.py', '-b', basedir, '-d', date, '-m', minute, '-p', path_to_file])