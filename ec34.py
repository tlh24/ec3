import torch as th
from torch import nn, optim
import torch.cuda.amp
import time
import argparse
import os
from data_exchange import SocketClient, MemmapOrchestrator
# from recognizer.model import Recognizer
from lifter.model import Lifter
# import wandb


parser = argparse.ArgumentParser(description='Transformer-based program synthesizer')
parser.add_argument("-b", "--batch_size", help="Set the batch size", type=int, default=32)
parser.add_argument("-d", "--dreaming", help="Set the model to dream", action="store_true")
args = parser.parse_args()
batch_size = args.batch_size
dreaming = args.dreaming
training = not dreaming
mmapno = 1 if dreaming else 0


print(f"batch_size:{batch_size}")
print(f"dreaming:{dreaming}")

# setup the socket client and handshake with the server.
socket_client = SocketClient(dreaming)
socket_client.connect()
# socket_client.handshake() -- not needed anymore!  April 27 2023

	
mo = MemmapOrchestrator(
	mmapno=mmapno,
	p_ctx=96,
	batch_size=batch_size,
	image_res=30
)

torch_device = 0
print("torch cuda devices", th.cuda.device_count())
print("torch device", th.cuda.get_device_name(torch_device))
th.cuda.set_device(torch_device)
# th.set_default_dtype('torch.float32')
th.set_default_device('cuda')
th.set_float32_matmul_precision('high') # desktop.


model = Lifter(use_l1=False, use_amort=False)

model.count_params()

from os.path import exists
fname = "checkpoints/lifter_checkpoint.ptx"
if exists(fname): 
	loaded_dict = torch.load(fname)
	model.load_state_dict(loaded_dict)
	print(f"loaded {fname}")
else: 
	if exists("ec34.ptx"):
		loaded_dict = torch.load("ec34.ptx")
		model.load_state_dict(loaded_dict)

model.to('cuda')

	
scaler = torch.amp.GradScaler('cuda')
slowloss = 1.0
losslog = open("loss_log.txt", "w")
tic = time.time()
if training:
	print("training...")
if dreaming:
	print("dreaming...")
	torch. set_grad_enabled(False)

# compiling this does not seem to work... 
# def train(mod, bimg, bpro, bedts):
# 	model.zero_grad()
# 	y,q = model(u, bimg, bpro)
# 	loss = lossfunc_mse(y, bedts)
# 	lossflat = th.sum(loss)
# 	lossflat.backward()
# 	th.nn.utils.clip_grad_norm_(model.parameters(), 0.025)
# 	optimizer.step()
# 	return y,q,lossflat
	

for u in range(100000):
	# keep things synchronous for now. 
	socket_client.send_and_receive(message="update_batch")
	
	bpro = mo.read_bpro()
	bimg = mo.read_bimg()
	
	if training:
		model.zero_grad()
		x = bimg.cuda()
		# add some noise to discourage sensitivity
		x = x + th.poisson(th.ones_like(x)) / 12 # perceptually, looks ok.
		lossdict = model.train_step(bimg.cuda(), bpro.cuda())
 
		weight_loss = lossdict["weight_loss"]
		ilv_loss = lossdict["ilv_loss"]
		amort_loss = lossdict["amort_loss"]
		losslog.write(f"{u}\t{weight_loss}\t{ilv_loss}\t{amort_loss}")
		losslog.write("\n")
		losslog.flush()
	
	# mo.write_bedtd(y)
	# if training:
	# 	mo.write_editdiff(bedts - y.cpu()) # synchronization.
	# socket_client.send_and_receive(message="decode_edit")
  
	if u % 11 == 0 or True:
		toc = time.time()
		rate = int((batch_size * 11) / (toc - tic))
		tic = toc
		print(f'{u} weight_loss: {weight_loss:.5f}; ilv_loss {ilv_loss:.5f}; {rate} samp/sec')
				
	# if u % 1000 == 999 :
	# 	if training:
	# 		model.save_checkpoint()
	# 	if dreaming:
	# 		model.load_checkpoint()
	# 		print("dreamer reloaded model parameters.")
	

mo.close()
# socket_client.close() called automatically

