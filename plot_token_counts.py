import pickle
import matplotlib.pyplot as plt
import sys
import os

def load_data(file_path):
    with open(file_path, 'rb') as file:
        while True:
            try:
                yield pickle.load(file)
            except EOFError:
                break

def plot_token_counts(file_path, layer, output_path):
    data = list(load_data(file_path))
    
    first_entry = next((entry for entry in data if entry[1] == layer), None)
    if not first_entry:
        print(f"No data found for layer {layer}")
        return
    
    # Determine if we have a single list or a list of lists
    is_double_list = isinstance(first_entry[2][0], list)

    if not is_double_list:
        iterations = [entry[0] for entry in data if entry[1] == layer]
        counts = [entry[2] for entry in data if entry[1] == layer]
        num_exp = len(counts[0])
        counts_per_token = list(map(list, zip(*counts)))

        plt.figure(figsize=(15, 10)) 
        plt.stackplot(iterations, *counts_per_token, edgecolor='black')
        plt.title(f'Num. Experts: {num_exp}. Layer: {layer}.', fontsize=40)
        plt.xlabel('Iteration', fontsize=36)
        plt.ylabel('# Tokens', fontsize=36)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)

        plt.savefig(output_path)
        plt.close()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(28, 10))
        
        for idx, ax in enumerate(axs):
            counts = [entry[2][idx] for entry in data if entry[1] == layer]
            counts_per_token = list(map(list, zip(*counts)))
            iterations = [entry[0] for entry in data if entry[1] == layer]

            ax.stackplot(iterations, *counts_per_token, edgecolor='black')
            prob_title = "1st prob." if idx == 0 else "2nd prob."
            ax.set_title(f'Num. Experts: {len(counts[0])}. Layer: {layer}. ({prob_title})', fontsize=40)
            ax.set_xlabel('Iteration', fontsize=36)
            ax.set_ylabel('# Tokens', fontsize=36)
            ax.tick_params(labelsize=14)
            plt.xticks(fontsize=26)
            plt.yticks(fontsize=26)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py token_counts.pkl layer output_path")
        sys.exit(1)

    file_path = sys.argv[1]
    layer = int(sys.argv[2])
    output_path = sys.argv[3]

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    plot_token_counts(file_path, layer, output_path)
