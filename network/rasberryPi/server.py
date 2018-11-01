import socket
import cv2
import numpy

def recvall(sock, count):
    buf = b''