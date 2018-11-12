import socket

TCP_IP = '192.168.0.105'
TCP_PORT = 5001

s = socket.socket()
s.connect((TCP_IP, TCP_PORT))

h = bytes("hello", "utf-8")

s.send(h)

h = bytes("quit", "utf-8")
s.send(h)
s.close()