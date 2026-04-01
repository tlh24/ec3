import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

def generate_independent_cayley(n=19):
	"""Generates a Cayley table where A, B, and C have independent random permutations."""
	# 1. Create three independent secret identities
	pi_a = np.random.permutation(n)
	pi_b = np.random.permutation(n)
	pi_c = np.random.permutation(n)

	# Inverse lookups to do the actual math
	inv_pi_a = np.argsort(pi_a)
	inv_pi_b = np.argsort(pi_b)

	# 2. Build the table of observed labels
	C_table = np.zeros((n, n), dtype=int)
	for i in range(n):     # i is the label of 'a'
		for j in range(n): # j is the label of 'b'
			actual_a = inv_pi_a[i]
			actual_b = inv_pi_b[j]

			# The actual group operation
			actual_c = (actual_a + actual_b) % n

			# Record the resulting label
			C_table[i, j] = pi_c[actual_c]

	return C_table

def extract_topology(W):
	"""
	Uses Spectral Graph Theory (Laplacian Eigenmaps) to infer 1D cyclic
	order from a cyclic shift Adjacency Matrix W.
	"""
	# 1. Compute the Graph Laplacian
	degree_matrix = np.diag(W.sum(axis=1))
	L = degree_matrix - W

	# 2. Eigen-decomposition
	# eigh is for symmetric matrices. It returns sorted eigenvalues.
	evals, evecs = eigh(L)

	# The smallest eigenvalue (index 0) is trivial (value 0, constant eigenvector).
	# The 2nd and 3rd eigenvectors (indices 1 and 2) embed the graph into
	# a 2D space. Because the graph is a cycle, this embedding is a perfect circle!
	x = evecs[:, 1]
	y = evecs[:, 2]

	# 3. Read the order of elements around the circle
	angles = np.arctan2(y, x)

	# Returning the sorted indices recovers the original topology
	return np.argsort(angles)

def unpermute_spectral(C_table, n=19):
	"""Recovers the ordering for A, B, and C purely from observation."""

	# --- 1. Un-permute B ---
	# Pick two random A labels to act as our shift operators = generator
	a0, a1 = 0, 1
	P_a0 = np.zeros((n, n))
	P_a1 = np.zeros((n, n))
	for b in range(n):
		P_a0[b, C_table[a0, b]] = 1
		# P_a0 maps b -> c given a0: P_a0 * b = c0
		P_a1[b, C_table[a1, b]] = 1
		# P_a1 maps b -> c given a1: P_a1 * b = c1

	# P_a0 @ P_a1.T connects elements in B that share the same shift.
	# Because n=19 is prime, ANY shift generates the full cycle.
	W_B = P_a0 @ P_a1.T
	# transpose is the inverse map c1 -> b given a1
	# @ traces a path b -> c -> b'
	# algebraically, a0 + b = c ; a1 + b' = c
	# then a0 - a1 = delta = b - b'
	# so W_B is 1 only where (b - b') mod n = delta (a const)
	# this defines neighbors in terms of one generator "delta". 
	W_B = W_B + W_B.T # Make W_B undirected/symmetric
	order_B = extract_topology(W_B)

	# --- 2. Un-permute A ---
	# Do the same, but using two B elements as operators
	b0, b1 = 0, 1
	P_b0 = np.zeros((n, n))
	P_b1 = np.zeros((n, n))
	for a in range(n):
		P_b0[a, C_table[a, b0]] = 1
		P_b1[a, C_table[a, b1]] = 1

	W_A = P_b0 @ P_b1.T
	W_A = W_A + W_A.T
	order_A = extract_topology(W_A)

	# --- 3. Un-permute C ---
	# Map C to C using A operators.
	W_C = P_a0.T @ P_a1
	W_C = W_C + W_C.T
	order_C = extract_topology(W_C)

	return order_A, order_B, order_C

def unpermute_aligned(C_table, n=19):
	"""Recovers and ALIGNS the orderings so A, B, and C share the same generator."""

	# 1. Extract a topology for B using two arbitrary A operators
	# (see more detailed comments above)
	a0, a1 = 0, 1
	P_a0, P_a1 = np.zeros((n, n)), np.zeros((n, n))
	for b_label in range(n):
		P_a0[b_label, C_table[a0, b_label]] = 1
		P_a1[b_label, C_table[a1, b_label]] = 1

	W_B = P_a0 @ P_a1.T
	W_B = W_B + W_B.T
	order_B = extract_topology(W_B) # B is now ordered by step size delta.

	# 2. ANCHOR C TO B
	# Pick an arbitrary row (e.g., the label 0).
	# If we pass our sorted B through this row, it produces a sequence of C labels.
	# By definition, this sequence shares the exact same delta as B!
	anchor_a = 0
	order_C = [C_table[anchor_a, b] for b in order_B]

	# 3. ANCHOR A TO C (and thus B)
	# How much does a row 'a' shift the sequence compared to our anchor?
	order_A = np.zeros(n, dtype=int)
	for a_label in range(n):
		# Look at the C output when 'a' interacts with the very first element of sorted B
		c_output = C_table[a_label, order_B[0]]

		# Where does this output sit in our canonical C sequence?
		# That index is exactly the topological position of 'a'!
		topological_idx = order_C.index(c_output)
		order_A[topological_idx] = a_label

	return order_A, order_B, order_C

def plot_classic_vs_recovered(n=19):
	# 1. Generate the scrambled data
	print("Generating independently permuted data...")
	C_table = generate_independent_cayley(n)

	# 2. Apply Spectral Graph Theory to find the hidden rings
	print("Calculating Graph Laplacians and decomposing...")
	order_A, order_B, order_C = unpermute_aligned(C_table, n)

	# 3. Apply the recovered orders to the table
	# Rearrange rows (A) and columns (B)
	recovered_table = C_table[order_A, :][:, order_B]

	# C is tricky: order_C tells us the sorted sequence of C labels.
	# We must map the observed arbitrary C label -> its new topological index (0 to 58)
	c_map = {old_label: new_idx for new_idx, old_label in enumerate(order_C)}

	mapped_recovered_table = np.zeros_like(recovered_table)
	for i in range(n):
		for j in range(n):
			mapped_recovered_table[i, j] = c_map[recovered_table[i, j]]

	# 4. Visualization
	fig, axes = plt.subplots(1, 2, figsize=(14, 7))

	im1 = axes[0].imshow(C_table, cmap='viridis', interpolation='nearest')
	axes[0].set_title('Observed C-Table\n(Independent Permutations pi_a,pi_b, pi_c)', fontsize=14)
	axes[0].set_xlabel('Label of b', fontsize=12)
	axes[0].set_ylabel('Label of a', fontsize=12)

	im2 = axes[1].imshow(mapped_recovered_table, cmap='viridis', interpolation='nearest')
	axes[1].set_title('Recovered Structure\n(via Laplacian Eigenmaps)', fontsize=14)
	axes[1].set_xlabel('Recovered Topological Index of b', fontsize=12)
	axes[1].set_ylabel('Recovered Topological Index of a', fontsize=12)

	plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="C Label")
	plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Recovered C Index")

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
    plot_classic_vs_recovered(19)
