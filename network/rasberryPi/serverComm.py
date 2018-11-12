import socket
import os
import sysconfig

TCP_IP = '192.168.0.105'
TCP_PORT = 5001

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
conn, addr = s.accept()
print("got connected from ", addr)


message = conn.recv(10)
secondMessage = conn.recv(10)

print(message, " ", secondMessage)
s.close()

