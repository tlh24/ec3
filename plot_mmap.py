import numpy as np
import torch as th
import argparse
import matplotlib.pyplot as plt
import os
from data_exchange import MemmapOrchestrator

from constants import *
# remove menubar buttons
plt.rcParams['toolbar'] = 'None'

parser = argparse.ArgumentParser(description='image mmaped files')
parser.add_argument("-b", "--batch_size", help="Set the batch size", type=int)
parser.add_argument("-d", "--dreaming", help="Set the model to dream", action="store_true")
parser.add_argument("-c", "--cycle", help="Cycle through the batch index", action="store_true")
args = parser.parse_args()
batch_size = args.batch_size
g_dreaming = args.dreaming
filno = 1 if args.dreaming else 0
print(f"batch_size:{batch_size}")

mo = MemmapOrchestrator(filno, p_ctx, batch_size, image_res)


plot_rows = 2
plot_cols = 3
figsize = (16, 9)
plt.ion()
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 72
fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize)
initialized = False
im = [ [0]*plot_cols for i in range(plot_rows)]
cbar = [ [0]*plot_cols for i in range(plot_rows)]
ov = [ [None]*plot_cols for i in range(plot_rows)]


def plot_tensor(r, c, v, name, lo, hi):
	if not initialized:
		# seed with random data so we get the range right
		cmap_name = 'PuRd' # purple-red
		if lo == -1*hi:
			cmap_name = 'seismic'
		data = np.linspace(lo, hi, v.shape[0] * v.shape[1])
		data = np.reshape(data, (v.shape[0], v.shape[1]))
		im[r][c] = axs[r,c].imshow(data, cmap = cmap_name)
		cbar[r][c] = plt.colorbar(im[r][c], ax=axs[r,c])
	im[r][c].set_data(v.numpy())
	#cbar[r][c].update_normal(im[r][c]) # probably does nothing
	axs[r,c].set_title(name)

def plot_line(r, c, v, name):
	axs[r,c].cla()
	axs[r,c].plot(v, 'bo')
	axs[r,c].set_title(name)

bs = batch_size
if batch_size > 32: 
	bs = 32

b = 0
while True:
	bpro     = mo.read_bpro()
	bpro_hold = mo.read_bpro_hold()
	bimg     = mo.read_bimg()
	logits   = mo.read_logits()
	posenc     = mo.read_posenc()
	bimg_recon = mo.read_bimg_recon()

	plot_line(0, 0, bpro[b,:], f"bpro[{b},:,:]")
	img0 = bimg[b,0,:,:] # + np.random.poisson(1, [image_res, image_res]) / 8
	img1 = bimg[b,1,:,:] # + np.random.poisson(1, [image_res, image_res]) / 8
	plot_tensor(0, 1, img0, f"bimg[{b},0,:,:]", -1.0, 1.0)
	plot_tensor(0, 2, img1, f"bimg[{b},1,:,:]", -1.0, 1.0)
	plot_tensor(1, 0, logits[b,:, :30].T, f"model logits", 0.0, 1.0)
	prog_slice = bpro_hold[b, p_ctx//2:].numpy()
	xs = np.arange(len(prog_slice))
	if ov[1][0] is None:
		ov[1][0], = axs[1,0].plot(xs, prog_slice, 'o', markersize=6,
			markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.0)
	else:
		ov[1][0].set_data(xs, prog_slice)
	plot_tensor(1, 2, bimg_recon[b,:,:], f"bimg_b_recon[{b}]", -1.0, 1.0)
	
	fig.tight_layout()
	fig.canvas.draw()
	fig.canvas.flush_events()
	# time.sleep(2)
	print("tick")
	initialized=True
	if args.cycle:
		b = (b+1) % bs

