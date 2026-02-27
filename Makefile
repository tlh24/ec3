
.PHONY: all clean force_lib

all: cuda_simsearch/libsimsearch.so data/t10k-labels-idx1-ubyte
	dune build

cuda_simsearch/libsimsearch.so: force_lib
	$(MAKE) -C cuda_simsearch

data/t10k-labels-idx1-ubyte:
	mkdir -p data
	cd data && \
	wget https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz && \
	wget https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz && \
	wget https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz && \
	wget https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz && \
	gunzip *.gz

clean:
	dune clean
	rm -f *.S
	rm -f *.gexf
