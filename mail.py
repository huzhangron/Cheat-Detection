#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:54:34 2020

@author: sanjanasrinivasareddy
"""

# Python code to illustrate Sending mail from  
# your Gmail account  
import smtplib 
  
# creates SMTP session 
s = smtplib.SMTP('smtp.gmail.com', 587) 
  
# start TLS for security 
s.starttls() 
  
# Authentication 
s.login("cheat.devp@gmail.com", "qwerty@123456") 
  
# message to be sent 
message = ""
  
# sending the mail 
s.sendmail("cheat.devp@gmail.com", "nishchaljs@gmail.com", message) 
  
# terminating the session 
s.quit() 