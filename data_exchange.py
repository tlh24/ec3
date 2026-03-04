import socket
import mmap
import math
import os
import torch as th
from ctypes import c_char
from functools import partial
from constants import e_indim, poslen

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


# Create a memory-mapped file, fallocating it first if it doesn't exist.
# dims is a list of float32 tensor dimensions used to compute the byte size.
def make_mmf(fname, dims):
	nbytes = math.prod(dims) * 4  # float32
	if not os.path.exists(fname):
		os.system(f'fallocate -l {nbytes} {fname}')
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

		self.fd_bpro      = make_mmf(f"bpro_{mmapno}.mmap",\
			[batch_size, p_ctx])
		self.fd_bpro_hold = make_mmf(f"bpro_hold_{mmapno}.mmap",\
			[batch_size, p_ctx])
		self.fd_bimg      = make_mmf(f"bimg_{mmapno}.mmap",\
			[batch_size, 2, image_res, image_res])
		self.fd_logits    = make_mmf(f"logits_{mmapno}.mmap",\
			[batch_size, p_ctx // 2, 64])
		self.fd_bedtd     = make_mmf(f"bedtd_{mmapno}.mmap",\
			[batch_size, e_indim])
		self.fd_posenc    = make_mmf(f"posenc_{mmapno}.mmap",\
			[p_ctx, poslen])
		self.fd_editdiff  = make_mmf(f"editdiff_{mmapno}.mmap",\
			[batch_size, e_indim])

		# Create partial functions for reading/writing with fixed shapes
		self.read_bpro       = partial(self.read_mmap, self.fd_bpro,
		                               [batch_size, p_ctx])
		self.read_bpro_hold  = partial(self.read_mmap, self.fd_bpro_hold,
		                               [batch_size, p_ctx])
		self.write_bpro_hold = partial(self.write_mmap, self.fd_bpro_hold)
		self.read_bimg       = partial(self.read_mmap, self.fd_bimg,
		                               [batch_size, 2, image_res, image_res])
		# logits: model outputs prog_B tokens only, so shape is [B, p_ctx//2, 64]
		self.read_logits     = partial(self.read_mmap, self.fd_logits,
		                               [batch_size, p_ctx // 2, 64])
		self.write_logits    = partial(self.write_mmap, self.fd_logits)
		self.read_bedtd      = partial(self.read_mmap, self.fd_bedtd,
		                               [batch_size, e_indim])
		self.read_posenc     = partial(self.read_mmap, self.fd_posenc,
		                               [p_ctx, poslen])
		self.read_editdiff   = partial(self.read_mmap, self.fd_editdiff,
		                               [batch_size, e_indim])

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
		self.fd_bpro_hold.close()
		self.fd_bimg.close()
		self.fd_logits.close()
		self.fd_bedtd.close()
		self.fd_posenc.close()
		self.fd_editdiff.close()
