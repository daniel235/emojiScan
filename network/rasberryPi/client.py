import socket
import cv2
import numpy

TCP_IP = 'localhost'
TCP_PORT = 5001

sock = socket.socket()
sock.connect(('192.168.0.105', TCP_PORT))
'''
capture = cv2.VideoCapture(0)
ret, frame = capture.read()

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
result, imgencode = cv2.imencode('.jpg', frame, encode_param)

data = numpy.array(imgencode)
stringData = data.tostring()

sock.send(str(len(stringData)).ljust(16).encode('utf-8'))
sock.send(stringData)'''
sock.close()

'''
decimg = cv2.imdecode(data, 1)
cv2.imshow('CLIENT', decimg)
cv2.waitKey(0)
cv2.destroyAllWindows()'''



