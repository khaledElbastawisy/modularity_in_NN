import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import networkx as nx
import tensorflow as tf 

def visualize_P_results(dict_neuron_share_ratio, datasetName, P_val):
    
    #track all other layers except dense and conv2D so we discard them while plotting
    if (datasetName == 'mnist'):
        other_layers= [2, 3, 6, 7, 8, 10]
    elif (datasetName == 'cifar10'):
        other_layers= [1,3,4,5,7,9,10,11,13,15,16,17,18,20]
    else:
        print('Wrong dataset name in visualize_P_results.')
        return 
    
    ylabel= 'Proportion'
    xlabel= 'Layer No.'
    #remove other layers
    dict_neuron_share_ratio = {k: v for k, v in dict_neuron_share_ratio.items() if k not in other_layers}
    
    layer_nums= list(dict_neuron_share_ratio.keys())
    ratios= list(dict_neuron_share_ratio.values())
    plt.figure(figsize=(10, 5))
    plt.bar(layer_nums, ratios)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0,1.0)
    plt.xticks(layer_nums)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if (P_val == 1): spec_or_reuse= 'spec'
    else: spec_or_reuse= 'reuse'
    plt.savefig(f"P_{spec_or_reuse}_{datasetName}.png")
    plt.show()
    plt.close()
    #exit()


def visualize_ablation_results(ablation_results):
    """
    Visualizes the results of the ablation studies.
    
    Parameters:
        ablation_results: Dictionary mapping removed class to evaluation results
    """
    # Extract data for visualization
    classes = list(ablation_results.keys())
    num_classes = len(classes)
    
    # Setup for multiple plots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Plot accuracy changes for each class when removing different subnetworks
    for removed_class in classes:
        acc_changes = []
        for cls in range(num_classes):
            change = ablation_results[removed_class]["accuracy_changes"].get(cls, 0)
            acc_changes.append(change)
        
        axs[0, 0].plot(range(num_classes), acc_changes, 'o-', label=f"Removed class {removed_class}")
    
    axs[0, 0].set_title("Effect on Class Accuracy after Subnetwork Removal")
    axs[0, 0].set_xlabel("Class")
    axs[0, 0].set_ylabel("Accuracy Change")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # 2. Plot heatmap of accuracy changes (class x removed subnetwork)
    change_matrix = np.zeros((num_classes, num_classes))
    for i, removed_class in enumerate(classes):
        for j, cls in enumerate(range(num_classes)):
            change = ablation_results[removed_class]["accuracy_changes"].get(cls, 0)
            change_matrix[j, i] = change
    
    im = axs[0, 1].imshow(change_matrix, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    axs[0, 1].set_title("Accuracy Change Heatmap")
    axs[0, 1].set_xlabel("Removed Subnetwork")
    axs[0, 1].set_ylabel("Affected Class")
    plt.colorbar(im, ax=axs[0, 1])
    
    # 3. Plot the effect on removed class itself
    removed_effects = [ablation_results[cls]["removed_class_effect"] for cls in classes]
    axs[1, 0].bar(classes, removed_effects)
    axs[1, 0].set_title("Effect on Removed Class Recognition")
    axs[1, 0].set_xlabel("Removed Class")
    axs[1, 0].set_ylabel("Accuracy Change")
    axs[1, 0].grid(True)
    
    plt.tight_layout()
    plt.savefig("ablation_results.png")
    plt.close()

def visualize_confusion_matrices(ablation_results, dataset='cifar10'):
    """
    Creates a grid of confusion matrices for each class ablation.
    
    Parameters:
        ablation_results: Dictionary mapping removed class to evaluation results
        dataset: Dataset name to determine class labels ('cifar10' or 'mnist')
    """
    # Define class names for the datasets
    if dataset == 'cifar10':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    else:  # mnist
        class_names = [str(i) for i in range(10)]
    
    # Extract data for visualization
    classes = list(ablation_results.keys())
    num_classes = len(classes)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_classes)))
    
    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    
    # Flatten the axes array for easy indexing
    axs = axs.flatten()
    
    # Plot confusion matrix for each class ablation
    for i, cls in enumerate(classes):
        if cls in ablation_results:
            # Get the confusion matrix
            confusion = ablation_results[cls]["confusion_matrix"]
            
            # Plot the confusion matrix
            im = axs[i].imshow(confusion, cmap='Blues', vmin=0, vmax=1)
            axs[i].set_title(f"Removed: {class_names[cls]}")
            
            # Only add x and y labels for subplots on the left and bottom edges
            if i % grid_size == 0:  # Left edge
                axs[i].set_ylabel("True Class")
            if i >= (grid_size * (grid_size - 1)):  # Bottom edge
                axs[i].set_xlabel("Predicted Class")
            
            # Set ticks and labels
            axs[i].set_xticks(range(num_classes))
            axs[i].set_yticks(range(num_classes))
            
            # Use smaller font size for tick labels
            axs[i].set_xticklabels([class_names[j] for j in range(num_classes)], rotation=90, fontsize=8)
            axs[i].set_yticklabels([class_names[j] for j in range(num_classes)], fontsize=8)
            
            # Add text annotations to the confusion matrix
            for row in range(num_classes):
                for col in range(num_classes):
                    # Conditionally show text based on value to avoid clutter
                    if confusion[row, col] > 0.1:  # Only show significant values
                        axs[i].text(col, row, f"{confusion[row, col]:.2f}",
                                   ha="center", va="center", fontsize=6,
                                   color="black" if confusion[row, col] < 0.5 else "white")
    
    # Hide any unused subplots
    for j in range(i + 1, grid_size * grid_size):
        axs[j].axis('off')
    
    # Add a colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Probability")
    
    # Add a main title
    fig.suptitle("Confusion Matrices for Each Class Ablation", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig("class_ablation_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_cm_diff(ablation_results, dataset='cifar10'):
    #visualize difference in CM after ablation
     # Define class names for the datasets
    if dataset == 'cifar10':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    else:  # mnist
        class_names = [str(i) for i in range(10)]
    
    # Extract data for visualization
    classes = list(ablation_results.keys())
    num_classes = len(classes)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_classes)))
    
    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    
    # Flatten the axes array for easy indexing
    axs = axs.flatten()
    
    # Plot confusion matrix for each class ablation
    for i, cls in enumerate(classes):
        if cls in ablation_results:
            confusion = ablation_results[cls]["confusion_matrix"]

            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion, annot=True, fmt='d', cmap='coolwarm',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title(f"Removed: {class_names[cls]}")
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.show()


#############################
# Graph Building Functions (Unchanged)
#############################
def get_activation_stats(neuron_stats):
    """
    Aggregates overall activation statistics across all layers.
    """
    all_means = []
    all_std = []
    for stats in neuron_stats.values():
        all_means.extend(stats['mean'])
        all_std.extend(stats['std'])
    if not all_means:
        raise ValueError("No neurons were processed to compute statistics.")
    return {
        'min_mean': np.min(all_means),
        'max_mean': np.max(all_means),
        'median_mean': np.median(all_means),
        'mean_of_means': np.mean(all_means),
        'min_std': np.min(all_std),
        'max_std': np.max(all_std)
    }

def get_neuron_ids(layer_index, layer, neuron_masks):
    """
    Returns unique node IDs for each neuron in the layer.
    """
    if layer_index in neuron_masks:
        n = neuron_masks[layer_index].shape[0]
        return [f"{layer_index}_{i}" for i in range(n)]
    else:
        if hasattr(layer, 'output_shape') and layer.output_shape is not None:
            shape = layer.output_shape
            if len(shape) == 2:
                n = shape[1]
            elif len(shape) == 4:
                n = shape[-1]
            else:
                n = shape[-1] if len(shape) > 1 else 1
        else:
            n = 1
        return [f"{layer_index}_{j}" for j in range(n)]

def build_graph_all_layers(model, neuron_masks, layer_activation_stats=None):
    """
    Builds a directed graph representation of the (sub)network.
    """
    G = nx.DiGraph()
    layer_nodes = {}
    for i, layer in enumerate(model.layers):
        node_ids = get_neuron_ids(i, layer, neuron_masks)
        for node in node_ids:
            neuron_index = int(node.split("_")[-1])
            G.add_node(node, layer=i, neuron_index=neuron_index, layer_type=type(layer).__name__)
        layer_nodes[i] = node_ids
    for i in range(1, len(model.layers)):
        prev_nodes = layer_nodes.get(i-1, [])
        curr_nodes = layer_nodes.get(i, [])
        curr_layer = model.layers[i]
        weights = None
        if hasattr(curr_layer, "get_weights"):
            w_list = curr_layer.get_weights()
            if w_list:
                weights = w_list[0]
        if weights is not None and isinstance(curr_layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            if isinstance(curr_layer, tf.keras.layers.Dense):
                for idx_prev, prev_node in enumerate(prev_nodes):
                    for idx_curr, curr_node in enumerate(curr_nodes):
                        try:
                            w_val = weights[idx_prev, idx_curr]
                        except IndexError:
                            w_val = 1.0
                        G.add_edge(prev_node, curr_node, weight=w_val)
            elif isinstance(curr_layer, tf.keras.layers.Conv2D):
                for idx_prev, prev_node in enumerate(prev_nodes):
                    for idx_curr, curr_node in enumerate(curr_nodes):
                        try:
                            w_val = np.mean(weights[:, :, idx_prev, idx_curr])
                        except IndexError:
                            w_val = 1.0
                        G.add_edge(prev_node, curr_node, weight=w_val)
        else:
            for prev_node in prev_nodes:
                for curr_node in curr_nodes:
                    G.add_edge(prev_node, curr_node, weight=1.0)
    return G

def build_combined_graph(subnetworks_info):
    """
    Combines subnetwork graphs for different classes into one graph.
    """
    G_combined = nx.DiGraph()
    for class_label, G_class in subnetworks_info:
        for node, node_data in G_class.nodes(data=True):
            new_node = f"{class_label}_{node}"
            node_data["class"] = class_label
            G_combined.add_node(new_node, **node_data)
        for source, target, edge_data in G_class.edges(data=True):
            new_source = f"{class_label}_{source}"
            new_target = f"{class_label}_{target}"
            G_combined.add_edge(new_source, new_target, **edge_data)
    return G_combined

def assign_positions(G, x_spacing=200, y_spacing=50):
    """
    Assigns (x, y) positions for each node for visualization.
    """
    layer_y_offsets = {}
    for node, data in G.nodes(data=True):
        layer = data.get('layer', 0)
        x = layer * x_spacing
        y_offset = layer_y_offsets.get(layer, 0)
        y = y_offset * y_spacing
        layer_y_offsets[layer] = y_offset + 1
        data['viz'] = {'position': {'x': float(x), 'y': float(y), 'z': 0.0}}
    return G

