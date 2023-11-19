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
    
    # Check the first entry to determine the structure of the data
    first_entry = next((entry for entry in data if entry[1] == layer), None)
    if not first_entry:
        print(f"No data found for layer {layer}")
        return
    
    # Determine if we have a single list or a list of lists
    is_double_list = isinstance(first_entry[2][0], list)

    # If we have a single list, plot as before
    if not is_double_list:
        iterations = [entry[0] for entry in data if entry[1] == layer]
        counts = [entry[2] for entry in data if entry[1] == layer]
        num_exp = len(counts[0])

        # Transpose the list of counts to have a list for each token index
        counts_per_token = list(map(list, zip(*counts)))

        plt.figure(figsize=(15, 10))  # Width, height in inches
        plt.stackplot(iterations, *counts_per_token, edgecolor='black')

        # Increase the size of the title and axis labels
        plt.title(f'Num. Experts: {num_exp}. Layer: {layer}.', fontsize=40)
        plt.xlabel('Iteration', fontsize=36)
        plt.ylabel('# Tokens', fontsize=36)

        # Set the size of the axis tick numbers
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)

        plt.savefig(output_path)
        plt.close()
    else:
        # If entry[2] is a list of two lists, create two subplots
        fig, axs = plt.subplots(1, 2, figsize=(28, 10))  # Two subplots in one column
        
        for idx, ax in enumerate(axs):
            # Extract counts for this subplot
            counts = [entry[2][idx] for entry in data if entry[1] == layer]
            counts_per_token = list(map(list, zip(*counts)))
            iterations = [entry[0] for entry in data if entry[1] == layer]

            ax.stackplot(iterations, *counts_per_token, edgecolor='black')

	    # Adjust the title based on the subplot index
            prob_title = "1st prob." if idx == 0 else "2nd prob."
            ax.set_title(f'Num. Experts: {len(counts[0])}. Layer: {layer}. ({prob_title})', fontsize=40)
            
            ax.set_xlabel('Iteration', fontsize=36)
            ax.set_ylabel('# Tokens', fontsize=36)
            ax.tick_params(labelsize=14)

            # Set the size of the axis tick numbers
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
