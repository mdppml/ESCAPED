import socket
import sys, time
import struct
import numpy as np
import pickle
import threading

class Connection:
	timeout = 10

	@staticmethod
	def startConnection(host, port):
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind((host, port))
		s.listen(1)
		print(f'Waiting for a connection... {host} - {port}')
		conn, addr = s.accept()
		print("Connection!!")
		# while True:
		# 	r = conn.recv(42).decode()
		# 	if r == "connected": break
		#
		# conn.sendall("<done>".encode())
		# time.sleep(0.5)

		print(f'Connection is established - {addr}')
		return (s, conn, addr)

	@staticmethod
	def connectHost(host, port):
		flag = False # indicating if the port is open
		while flag == False:
			s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			print(f'Checking the host to connect... {host} - {port}')
			flag = Connection.isOpen(s, host, port)
			time.sleep(0.5)

		#response = s.connect((host, port))
		# print("******* Response: {}".format(response))
		print('Flag become true')

		# s.sendall("connected".encode()) # the signal for checking the connection
		# time.sleep(0.5)
		#
		# while True:
		# 	r = s.recv(43).decode()
		# 	if r == "<done>": break

		# print("Connected to the host")
		return s

	@staticmethod
	def sendData(conn, data):
		out = pickle.dumps(data, -1)
		# print("Size of the data to send: {:.0f}".format(sys.getsizeof(out)))
		# tmp = format(sys.getsizeof(out), '010d') + ";"
		tmp = f'{sys.getsizeof(out)};'
		# print("tmp size: ", sys.getsizeof(tmp))
		at = tmp.encode()
		# print(sys.getsizeof(at))
		size_tmp = conn.send(at)
		# print("Send return: {:.0f}".format(size_tmp))
		while True:
			r = conn.recv(39).decode()
			if r == "<done>": break
		all_tmp = conn.sendall(out)
		# print("Sendall return: {}".format(all_tmp))
		response = ""
		while response != "<done>":
			response = conn.recv(39).decode()
		# print("Response is received: ", response)
		del out
		del tmp

	@staticmethod
	def receiveData(conn):
		read_max = 2**35
		# m_length = int(conn.recv(48).split(";")[0]) # the length of the data is sent in 10 character-length string
		while True:
			tmp = conn.recv(44).decode()
			if tmp: break
		m_length = int(tmp.split(";")[0]) # the length of the data is sent in 10 character-length string
		print("Size of the data to receive: {:.0f}".format(m_length))
		done_msg = "<done>".encode()
		# print("<done> size: ", sys.getsizeof(done_msg))
		conn.send(done_msg)
		# time.sleep(1)
		ultimate_buffer=b''
		while sys.getsizeof(ultimate_buffer) < m_length:
			to_read = m_length - sys.getsizeof(ultimate_buffer)
			receiving_buffer = conn.recv(read_max if to_read > read_max else to_read)
			#print(sys.getsizeof(receiving_buffer))
			if not receiving_buffer: break
			ultimate_buffer+= receiving_buffer
			#print(sys.getsizeof(ultimate_buffer))
			#print(len(ultimate_buffer))
			# print '-'
		# print("\nSize of the received data: {:.0f}".format(sys.getsizeof(ultimate_buffer)))
		conn.send(done_msg)
		# return np.load(StringIO(ultimate_buffer))['frame']
		return pickle.loads(ultimate_buffer)

	@staticmethod
	def isOpen(s, ip, port):
		try:
			s.connect((ip, int(port)))
			return True
		except socket.error as exc:
			# print(f'Caught exception socket.error: {exc}')
			return False





#atadam
