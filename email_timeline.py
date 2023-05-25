"""
Filename:   email_timeline.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Thu May 25 11:13:21 2023
Updated:    Thu May 25 11:13:21 2023
    
Usage: python email_timeline.py <path_to_timeline_file>
"""

# Module Imports
import os
import smtplib
from pathlib import Path
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate


#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = Path('D:')
RED_PATH  = Path('R:')
BLUE_PATH = Path('B:')
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'
LOG_PATH = BASE_PATH / 'Logs' / 'Operations'

# Email parameters
FROM_ADDRESS = ''
TO_ADDRESS = ['pquigle@uwo.ca']
SERVER_IP = '127.0.0.1'


#-------------------------------functions-------------------------------------#

def send_mail(send_from, send_to, subject, text, files=None, server=SERVER_IP):
    
    
    assert isinstance(send_to, list)

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = f"attachment; filemane={f.name}"
        msg.attach(part)


    smtp = smtplib.SMTP(server)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()


#---------------------------------main----------------------------------------#

if __name__ == '__main__':

#############################
## Argument Parser
#############################

    ## Generate argument parser
    parser = argparse.ArgumentParser(description='Email timeline file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Available argument functionality
    arg_parser.add_argument('date', help='Observation date (YYYY/MM/DD) of data to be processed')
