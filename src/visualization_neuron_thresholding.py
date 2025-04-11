import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import networkx as nx
import tensorflow as tf 

def visualize_cm_diff(cm_diff, dataset, rmvd_cls):
    #visualize difference in CM after ablation
     # Define class names for the datasets
    if dataset == 'cifar10':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    else:  # mnist
        class_names = [str(i) for i in range(10)]
    
    # Extract data for visualization
    #classes = list(ablation_results.keys())
    

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_diff, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Removed: {class_names[rmvd_cls]}")
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

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



#####       Graph Related Functions         #####


def get_activation_stats(important_neurons_activations):
    """
    Returns statistics for important neuron activations across all layers
    Returns: Dictionary with min, max, median, mean, and std of activations
    """
    all_activations = []
    
    # Flatten all activation values
    for layer_activations in important_neurons_activations.values():
        try:
            all_activations.extend(layer_activations.values())
        except:
            all_activations.extend(layer_activations)
        
    if not all_activations:
        raise ValueError("No important neurons found in the provided data")
    
    return {
        'min': np.min(all_activations),
        'max': np.max(all_activations),
        'median': np.median(all_activations),
        'mean': np.mean(all_activations),
        'std': np.std(all_activations),
        'total_neurons': len(all_activations)
    }


def get_neuron_ids(layer_index, layer, important_neurons):
    """
    Return a list of node IDs for the layer.
    If the layer is in the important_neurons dict, use those indices.
    Otherwise, derive the number of neurons from the layer's output_shape.
    """
    # Use important neurons if provided
    if layer_index in important_neurons:
        indices = important_neurons[layer_index]
        return [f"{layer_index}_{i}" for i in indices]
    else:
        # Attempt to use the output shape
        if hasattr(layer, 'output_shape') and layer.output_shape is not None:
            shape = layer.output_shape
            # If shape is (None, features) use features; if (None, H, W, C), use C
            if len(shape) == 2:
                n = shape[1]
            elif len(shape) == 4:
                n = shape[-1]
            else:
                n = shape[-1] if len(shape) > 1 else 1
        else:
            n = 1  # fallback: one node if unknown
        return [f"{layer_index}_{j}" for j in range(n)]


def build_graph_all_layers(model, important_neurons, layer_activation_stats=None):
    """
    Build a directed graph for the subnetwork represented by `model`.
    Every layer (weighted or not) is assigned nodes and consecutive layers are connected.
    For weighted layers (Dense, Conv2D), the corresponding weight matrix is used.
    For other layers, a default edge weight (1.0) is assigned.
    """
    G = nx.DiGraph()
    layer_nodes = {}  # layer index -> list of node IDs

    # Create nodes for each layer
    for i, layer in enumerate(model.layers):
        node_ids = get_neuron_ids(i, layer, important_neurons)
        for node in node_ids:
            # Extract neuron index from the node ID (the part after the underscore)
            neuron_index = int(node.split("_")[-1])
            # Save some attributes including layer type
            G.add_node(node, layer=i, neuron_index=neuron_index, layer_type=type(layer).__name__)
        layer_nodes[i] = node_ids

    # Now add edges between consecutive layers
    for i in range(1, len(model.layers)):
        prev_nodes = layer_nodes.get(i-1, [])
        curr_nodes = layer_nodes.get(i, [])
        prev_layer = model.layers[i-1]
        curr_layer = model.layers[i]

        # Attempt to extract weights from the current layer if available
        weights = None
        if hasattr(curr_layer, "get_weights"):
            w_list = curr_layer.get_weights()
            if w_list:  # typically, first element is the weights
                weights = w_list[0]

        # For layers with weights (Dense, Conv2D) use them; otherwise, use default
        if weights is not None and isinstance(curr_layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            if isinstance(curr_layer, tf.keras.layers.Dense):
                # weights shape: (n_prev, n_curr)
                for idx_prev, prev_node in enumerate(prev_nodes):
                    for idx_curr, curr_node in enumerate(curr_nodes):
                        try:
                            w_val = weights[idx_prev, idx_curr]
                        except IndexError:
                            w_val = 1.0  # fallback if dimensions don't match
                        G.add_edge(prev_node, curr_node, weight=w_val)
            elif isinstance(curr_layer, tf.keras.layers.Conv2D):
                # weights shape: (kernel_height, kernel_width, n_prev, n_curr)
                for idx_prev, prev_node in enumerate(prev_nodes):
                    for idx_curr, curr_node in enumerate(curr_nodes):
                        try:
                            # Aggregate over the kernel spatial dimensions; using mean here
                            w_val = np.mean(weights[:, :, idx_prev, idx_curr])
                        except IndexError:
                            w_val = 1.0
                        G.add_edge(prev_node, curr_node, weight=w_val)
        else:
            # For layers without weights (e.g., pooling, flatten, activation), create full connectivity with default weight.
            for prev_node in prev_nodes:
                for curr_node in curr_nodes:
                    G.add_edge(prev_node, curr_node, weight=1.0)
    return G


def build_combined_graph(subnetworks_info):
    """
    Combine subnetwork graphs from different classes into one graph.
    
    subnetworks_info: list of tuples (class_label, G_class) for each class.
    """
    G_combined = nx.DiGraph()
    
    for class_label, G_class in subnetworks_info:
        # Iterate over nodes and add them with a class-specific prefix
        for node, node_data in G_class.nodes(data=True):
            # Prefix node id with the class label to avoid collisions
            new_node = f"{class_label}_{node}"
            # Optionally, add a 'class' attribute to keep track of the subnetwork
            node_data["class"] = class_label
            G_combined.add_node(new_node, **node_data)
        
        # Iterate over edges and update node names accordingly
        for source, target, edge_data in G_class.edges(data=True):
            new_source = f"{class_label}_{source}"
            new_target = f"{class_label}_{target}"
            G_combined.add_edge(new_source, new_target, **edge_data)
    
    return G_combined


def assign_positions(G, x_spacing=200, y_spacing=50):
    """
    Assigns positions to each node based on its 'layer' attribute.
    Nodes in the same layer get different y positions.
    
    The viz:position attribute is added to each node so that Gephi
    uses these coordinates when loading the GEXF file.
    """
    # Dictionary to track current y offset for each layer
    layer_y_offsets = {}
    
    for node, data in G.nodes(data=True):
        # Retrieve the layer number (default to 0 if missing)
        layer = data.get('layer', 0)
        
        # Calculate x coordinate based on layer number
        x = layer * x_spacing
        
        # Use a counter to set a unique y coordinate for each node in the same layer
        y_offset = layer_y_offsets.get(layer, 0)
        y = y_offset * y_spacing
        layer_y_offsets[layer] = y_offset + 1
        
        # Add viz position attribute; note the structure Gephi expects:
        # { "position": {"x": ..., "y": ..., "z": ...} }
        data['viz'] = {'position': {'x': float(x), 'y': float(y), 'z': 0.0}}
    
    return G
