import socket
import mmap
import torch as th
from ctypes import c_char
from functools import partial

class SocketClient:
	"""

	# Usage example:


	with SocketClient(dreaming=False) as client:
		data = client.send_and_receive(
			message="decode_edit", recv_size=100)
	"""
	def __init__(
		self,
		dreaming: bool,
		host: str = '127.0.0.1',
		):
		self.dreaming = dreaming
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.host = host

	@property
	def port(self,):
		return 4341 if self.dreaming else 4340

	def connect(self):
		return self.sock.connect((self.host, self.port))

	def __enter__(self):
		return self.connect()

	def send_and_receive(self, message: str, recv_size:int=100):
		self.sock.sendall(message.encode())
		data = self.sock.recv(recv_size)
		return data

	def handshake(self,):
		data = self.send_and_receive(message="update_batch", recv_size=1024)
		print(f"Received {data!r}"
				)

	def __exit__(self, exc_type, exc_value, traceback):
		self.sock.close()


# Function to create a memory-mapped file
def make_mmf(fname): 
	fd = open(fname, "r+b")
	return mmap.mmap(fd.fileno(), 0)



class MemmapOrchestrator(object):
	def __init__(
		self,
		mmapno: int,
		p_ctx : int,
		batch_size : int,
		image_res : int,
		):
		self.fd_bpro = make_mmf(f"bpro_{mmapno}.mmap")
		self.fd_bimg = make_mmf(f"bimg_{mmapno}.mmap")

		# Create partial functions for reading data with the specified dimensions
		self.read_bpro = partial(self.read_mmap, self.fd_bpro,
										[batch_size, p_ctx])
		self.read_bimg = partial(self.read_mmap, self.fd_bimg,
										[batch_size, 2, image_res, image_res])

	def read_mmap(self, mmf, dims):
		mmf.seek(0)
		mmb = mmf.read()
		siz = len(mmb)
		mmb2 = (c_char * siz).from_buffer_copy(mmb)
		x = th.frombuffer(mmb2, dtype=th.float).clone()
		x = th.reshape(x, dims)
		return x

	def write_mmap(self, mmf, data):
		q = data.detach().cpu().numpy().tobytes()
		mmf.seek(0)
		n = mmf.write(q)
		return n

	def close(self):
		self.fd_bpro.close()
		self.fd_bimg.close()
