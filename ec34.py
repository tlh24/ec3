import torch as th
from torch import nn, optim
import torch.nn.functional as F
import torch.cuda.amp
import time
import argparse
import os
from data_exchange import SocketClient, MemmapOrchestrator
# from recognizer.model import Recognizer
from lifter.model import ForwardGraphics, InverseGraphics, CombinedModel
import pdb


parser = argparse.ArgumentParser(description='Transformer-based program synthesizer')
parser.add_argument("-b", "--batch_size", help="Set the batch size", type=int, default=32)
parser.add_argument("-d", "--dreaming", help="Set the model to dream", action="store_true")
args = parser.parse_args()
batch_size = args.batch_size
dreaming = args.dreaming
training = not dreaming
mmapno = 0


print(f"batch_size:{batch_size}")
print(f"dreaming:{dreaming}")

# setup the socket client and handshake with the server.
socket_client = SocketClient(False)
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

inv = InverseGraphics()
fwd = ForwardGraphics()
model = CombinedModel(inv, fwd)

inv.count_params()
fwd.count_params()

from os.path import exists
if exists("ec34.pt"):
	model.load_checkpoint("ec34.pt")

model.to('cuda') # does this call to() on the internal models?

	
scaler = torch.amp.GradScaler('cuda')
slowloss = 1.0
losslog = open("loss_log.txt", "w")
tic = time.time()
if training:
	print("training...")
if dreaming:
	print("dreaming...")
	

for u in range(100000):
	# keep things synchronous for now. 
	socket_client.send_and_receive(message="update_batch")
	
	bpro = mo.read_bpro()
	bimg = mo.read_bimg()
	x = bimg.cuda()
	
	if training:
		model.zero_grad()
		# add some noise to discourage sensitivity
		x = x + th.poisson(th.ones_like(x)) / 20 # perceptually, looks ok.
		y, img_b_recon, lossdict = model.train_step(x, bpro.cuda())

		ce_loss = lossdict["ce_loss"]
		recon_loss = lossdict["recon_loss"]
		total_loss = lossdict["total_loss"]
		losslog.write(f"{u}\t{ce_loss}\t{recon_loss}")
		losslog.write("\n")
		losslog.flush()
	else:
		for k in range(10):
			y, img_b_recon, lossdict = model.predict(x, bpro.cuda(), k)
			mo.write_logits(F.softmax(y, -1))
			mo.write_bpro_hold(bpro)
			mo.write_bimg_recon(img_b_recon)
			socket_client.send_and_receive(message="decode_logits")
			time.sleep(0.5)
		ce_loss_pre = lossdict["ce_loss_pre"]
		ce_loss_post = lossdict["ce_loss_post"]
		losslog.write(f"{u}\t{ce_loss_pre}\t{ce_loss_post}")
		losslog.write("\n")
		losslog.flush()
	
	mo.write_logits(F.softmax(y, -1))
	mo.write_bpro_hold(bpro)
	mo.write_bimg_recon(img_b_recon)
	socket_client.send_and_receive(message="decode_logits")
  
	if u % 11 == 0 :
		toc = time.time()
		rate = int((batch_size * 11) / (toc - tic))
		tic = toc
		if training:
			print(f'{u} ce_loss: {ce_loss:.5f}; recon_loss {recon_loss:.5f}; {rate} samp/sec')
		else:
			print(f'{u} ce_loss_pre: {ce_loss_pre:.5f}; ce_loss_post {ce_loss_post:.5f}; {rate} samp/sec')
				
	if u % 100 == 99 :
		if training:
			model.save_checkpoint("ec34.pt")
		if dreaming:
			model.load_checkpoint("ec34.pt")
			print("dreamer reloaded model parameters.")
	

mo.close()
# socket_client.close() called automatically

