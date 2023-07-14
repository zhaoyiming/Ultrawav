from socket import *
import time
import os
import datetime

def file_transfer():
    while True:
        tcp_server = socket(AF_INET, SOCK_STREAM)
        address = ('', 7365)
        tcp_server.bind(address)
        tcp_server.listen(1)
        client_socket, clientAddr = tcp_server.accept()
        ttime = path + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.pcm'
        with open(ttime, 'wb') as outfile:
            while True:
                block = client_socket.recv(1024)
                if not block:
                    break
                outfile.write(block)
        tcp_server.close()
        print("succ")


if __name__ == '__main__':
    path = '/Users/zhaoyiming/Documents/matlab/receive/1013/'
    if not os.path.isdir(path):
        os.mkdir(path)
    file_transfer()
