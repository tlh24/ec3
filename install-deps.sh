# -- Commands for getting Lambda instance up and running --

# Install conda
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local && rm Miniconda3-latest-Linux-x86_64.sh && /usr/local/bin/conda init


sudo chmod +rw /usr/bin
sudo apt-get update
sudo apt-get install -y make gcc unzip bubblewrap libpcre2-dev menhir

# Create and activate the Conda environment
conda env create -f environment.yml
source activate ec3

# upgrade opam
opam init
opam update --confirm-level=unsafe-yes
opam switch create myswitch ocaml-variants.5.0.0+options ocaml-option-flambda libcairo2-dev
# flambda speeds up executution, at the cost of longer compilation.
eval $(opam env --switch=myswitch)
opam update --confirm-level=unsafe-yes

# need to install libtorch
wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu116.zip
mv libtorch-cxx11-abi-shared-with-deps-1.13.1+cu116.zip ~
unzip ~/libtorch-cxx11-abi-shared-with-deps-1.13.1+cu116.zip
mv libtorch ~
export LIBTORCH=~/libtorch

opam install --confirm-level=unsafe-yes vg cairo2 vector lwt logs domainslib ocamlgraph psq ctypes utop
# ocaml-torch presently needs to be built from source, alas.
eval $(opam env --switch=myswitch)
dune build

# for accessing remotely:  (e.g.)
# sshfs -o allow_other,default_permissions ubuntu@104.171.203.63:/home/ubuntu/cortex/ /home/tlh24/remote/
