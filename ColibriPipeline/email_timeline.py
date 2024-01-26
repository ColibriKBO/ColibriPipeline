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
import argparse
import smtplib
from smtplib import SMTP
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders


#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = Path('D:')
RED_PATH  = Path('R:')
BLUE_PATH = Path('B:')
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'
LOG_PATH = BASE_PATH / 'Logs' / 'Operations'

# Email parameters
FROM_ADDRESS = ''
TO_ADDRESS   = ['pquigle@uwo.ca']
SERVER_IP = '10.0.20.40'  # IP address of Greenbird
HOSTNAME  = 'localhost'


#-------------------------------functions-------------------------------------#

def sendMail(send_from, send_to, subject, message, files=[], 
              server=HOSTNAME, port=587, username='', password='',
              use_tls=True):
    """
    Send an email with attachments. Credit to Ehsan Iran-Nejad on StackOverflow:
    https://stackoverflow.com/questions/3362600/how-to-send-email-attachments
    
    Parameters
    ----------
    send_from : str
        Email address of sender
    send_to : list
        List of email addresses of recipients
    subject : str
        Subject line of email
    text : str
        Body of email
    files : list, optional
        List of file paths to attach to email
    server : str, optional
        IP address of SMTP server
    
    Returns
    -------
    None
    """
    
    assert isinstance(send_to, list)

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(message))

    # Attach files
    for path in files:
        part = MIMEBase('application', "octet-stream")
        with open(path, 'rb') as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename={}'.format(Path(path).name))
        msg.attach(part)

    # Initialize SMTP server
    smtp = smtplib.SMTP(server, port)
    if use_tls:
        smtp.starttls()
    
    # Login to server
    try:
        smtp.login(username, password)
    except SMTP.SMTPHeloError:
        print('Server did not reply')
        smtp.quit()
        return 1
    except SMTP.SMTPAuthenticationError:
        print('Incorrect username/password combination')
        smtp.quit()
        return 1
    
    # Send email
    try:
        smtp.sendmail(send_from, send_to, msg.as_string())
    except SMTP.SMTPRecipientsRefused:
        print('All recipients were refused')
        smtp.quit()
        return 1
    
    # Show email to user and close connection
    displayMail(send_from, send_to, subject, text, files)
    print('\n\nEmail sent successfully!')
    smtp.quit()




def displayMail(send_from, send_to, subject, text, files=[]):
    """
    Print email to console. 
    
    Parameters
    ----------
    send_from : str
        Email address of sender
    send_to : list
        List of email addresses of recipients
    subject : str
        Subject line of email
    text : str
        Body of email
    files : list, optional
        List of file paths to attach to email
    server : str, optional
        IP address of SMTP server
    
    Returns
    -------
    None
    """

    # Print email header
    print("\n" + "-"*50)
    print("From: {}".format(send_from))
    print("To: {}".format(send_to))
    print("Subject: {}".format(subject))
    print("-"*50 + "\n")

    # Print email body and attachments
    print("Text: {}\n".format(text))
    for f in files:
        print("Attached: {}".format(f.name))
    print("-"*50 + "\n")


#---------------------------------main----------------------------------------#

if __name__ == '__main__':

#############################
## Argument Parser
#############################

    # Generate argument parser
    arg_parser = argparse.ArgumentParser(description='Email timeline file',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add argument functionality
    arg_parser.add_argument('--sendto', type=list, default=TO_ADDRESS,
                            help='List of email addresses to send to.')
    arg_parser.add_argument('--subject', type=str, default='Colibri Alert!',
                            help='Subject line of email.')
    arg_parser.add_argument('--text', type=str, default='',
                            help='Body of email.')
    arg_parser.add_argument('--files', type=list, default=[],
                            help='List of file paths to attach to email.')
    arg_parser.add_argument('--dry', action='store_true',
                            help='Dry run. Do not send email.')
    
    # Parse arguments
    args = arg_parser.parse_args()
    to      = args.sendto
    subject = args.subject
    text    = args.text
    files   = args.files


#############################
## Send Email
#############################

    if not args.dry:
        sendMail(FROM_ADDRESS, to, subject, text, files)
    else:
        displayMail(FROM_ADDRESS, to, subject, text, files)
        print('\n\nDry run. Email not sent.')