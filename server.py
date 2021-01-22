#!coding=utf-8

import threading
import socket
import os
import struct

my_ip = "86.246.41.192"
open_port = 9001

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path.encode('utf-8')) 
        return True
    else:
        return False

def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((my_ip, open_port))
        s.listen(10)
    except socket.error as msg:
        print (msg)
        sys.exit(1)
    print ('Waiting connection...')
 
    while 1:
        conn, addr = s.accept()
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()
        
 
def deal_data(conn, addr):
    print ('Accept new connection from {0}'.format(addr))
    conn.send('Hi, Welcome to the server!'.encode('utf-8'))
 
    while 1:
        fileinfo_size = struct.calcsize('128sl')
        buf = conn.recv(fileinfo_size)
        if buf:
            filename, filesize = struct.unpack('128sl', buf)
            fn = filename.strip(b'\00')
            fn = fn.decode(errors="ignore")
            print ('file new name is {0}, filesize is {1}'.format(str(fn),filesize))
 
            recvd_size = 0  
            mkdir('./fichiers')
            fp = open('./fichiers/'+ str(fn), 'wb')
            print ('start receiving...')
            
            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = conn.recv(1024)
                    recvd_size += len(data)
                else:
                    data = conn.recv(filesize - recvd_size)
                    recvd_size = filesize
                fp.write(data)
            fp.close()
            print('end receive...')
        
        

if __name__ == "__main__":
    socket_service()

