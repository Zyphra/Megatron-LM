import pickle
import torch
import matplotlib.pyplot as plt
import sys
import os
import torch.nn.functional as F

def load_data(file_path):
    with open(file_path, 'rb') as file:
        while True:
            try:
                yield pickle.load(file)
            except EOFError:
                break

def plot_token_counts(file_path, iteration, start_expert, output_path):
    data = list(load_data(file_path))
    idx_list = [entry[2] for entry in data if entry[0] == iteration]

    if not idx_list:
        print(f"No data found for iteration {iteration}")
        return

    # Extracting counts
    E = max(max(sublist) for sublist in idx_list) + 1
    start_idx = idx_list[0]
    mask = (torch.tensor(start_idx) == start_expert).nonzero(as_tuple=True)[0]
    counts = []
    for idx in idx_list:
        idx_tensor = torch.tensor(idx)
        idx_sub = F.one_hot(idx_tensor[mask], num_classes=E)
        counts.append(torch.sum(idx_sub, dim=0))

    # Sorting counts in decreasing order
    counts_tensor = torch.stack([torch.sort(c, descending=True)[0] for c in counts])

    # Create the stackplot
    x_values = torch.arange(counts_tensor.shape[0])
    counts_transposed = counts_tensor.T
    plt.figure(figsize=(10, 10))
    plt.stackplot(x_values, *counts_transposed, edgecolor='black')
    plt.xlabel('Sorted Tensor Index')
    plt.ylabel('Magnitude')
    plt.title(f'Stackplot of Sorted Counts for Iteration {iteration}')

    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py token_counts.pkl iteration start_expert output_path")
        sys.exit(1)

    file_path = sys.argv[1]
    iteration = int(sys.argv[2])
    start_expert = int(sys.argv[3])
    output_path = sys.argv[4]

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    plot_token_counts(file_path, iteration, start_expert, output_path)
