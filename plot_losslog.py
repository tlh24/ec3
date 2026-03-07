import numpy as np
import csv
import pdb
import matplotlib.pyplot as plt
import time

# remove menubar buttons
plt.rcParams['toolbar'] = 'None'

plot_rows = 1
plot_cols = 2
figsize = (16, 9)
plt.ion()
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 72
fig, ax = plt.subplots(plot_rows, plot_cols, figsize=figsize)
initialized = False

while True: 
	with open("loss_log.txt", 'r') as x:
		data = list(csv.reader(x, delimiter="\t"))

	data = np.array(data)
	data = data.astype(float)

	ax[0].cla()
	ax[0].plot(data[:,0], np.log(data[:, 1]), 'k')
	ax[0].set(xlabel='iteration')
	ax[0].set_title('ce loss')

	ax[1].cla()
	ax[1].plot(data[:,0], np.log(data[:, 2]), 'k')
	ax[1].set(xlabel='iteration')
	ax[1].set_title('pixel mse loss')

	fig.tight_layout()
	fig.canvas.draw()
	fig.canvas.flush_events()
	time.sleep(1)
	print("tick")
	#plt.show()
