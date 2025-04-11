import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import dataloader
import argparse 
import networkx as nx
from sklearn.metrics import confusion_matrix
import random 
import visualization_neuron_thresholding as vis


def P_spec_and_P_reuse(original_model, x_test, y_test, datasetName, P_val):
    '''
    plot result for P_specialization test.
    x_test, y_test: whole test dataset.
    P_val=1 test for P_spec, 0: test for P_reuse
    '''
    num_repeat= 10      #how many repeated experiments we want to run?
    neuron_share_ratio= {}      #key: layer_number, value: avg ratio of shared neuron (avg over num_repeat tests)
    #initialize the dict
    for i in range (len(original_model.layers)):
        neuron_share_ratio[i]= 0

    for c in range (num_repeat):
        if (P_val == 1):        #test for P_spec
            classes= random.sample(range(10), 2)           #a list of size 2. Which two different classes do we want to experiment for?
            
            #subselect data for selected classes
            class_indices_1 = np.where(np.argmax(y_test, axis=1) == classes[0])[0]
            x_class_1 = x_test[class_indices_1]
            y_class_1= y_test[class_indices_1]

            class_indices_2 = np.where(np.argmax(y_test, axis=1) == classes[1])[0]
            x_class_2 = x_test[class_indices_2]
            y_class_2= y_test[class_indices_2]
        
        elif (P_val == 0):  #test for P_reuse
            chosen_class = random.choice(range(10))   #choose a class
            classes = [chosen_class, chosen_class]

            #subselect data for selected class
            class_indices = np.where(np.argmax(y_test, axis=1) == chosen_class)[0]
            x_class = x_test[class_indices]
            y_class= y_test[class_indices]

            #now divide the data of one class into two datasets
            half= len(x_class)//2
            x_class_1= x_class[0:half]
            y_class_1= y_class[0:half]
            x_class_2= x_class[half:]
            y_class_2= y_class[half:]
        else:
            print ('Wrong P_val, choos either 1 or 0')
            return   
        
        imp_neurons_1, _, _ = get_important_neurons(
            original_model, x_class_1, y_class_1, classes[0], 70
        )

        imp_neurons_2, _, _ = get_important_neurons(
            original_model, x_class_2, y_class_2, classes[1], 70
        )       
        
        #neuron_mask_i is a dictionary. {layer_num: array([-1,1,0,....])}
        for layer_num, imp_neuron_indices_1 in imp_neurons_1.items():
            imp_neuron_indices_2= imp_neurons_2[layer_num]
            #consider only ACTIVE neurons
            similar_masks = np.where(imp_neuron_indices_1 == imp_neuron_indices_2 , 1, 0)
            similarity_ratio= np.sum(similar_masks)/len(imp_neuron_indices_1)
            neuron_share_ratio[layer_num] = neuron_share_ratio[layer_num] + similarity_ratio

    #take average over num_repeat tests
    for i in range (len(original_model.layers)):
        neuron_share_ratio[i]= neuron_share_ratio[i]/num_repeat
    print(neuron_share_ratio)

    #Now plot the results
    vis.visualize_P_results(neuron_share_ratio, datasetName, P_val)


def create_sub_network(original_model, important_neurons):
    '''
    params:
    original_model: the true model, trained on all classes
    important_neurons: a dictionary that looks like {layer_index: important neuron indices}
    returns:
    a subnet/module consisting only the important neurons & their weights, biases. All other weights, biases are zeroed out.
    '''
    sub_network_layers = []
    for i, layer in enumerate(original_model.layers):
        if isinstance(layer, layers.Conv2D):
            # Get weights and biases from original layer
            weights, biases = layer.get_weights()
            
            # Apply mask=1 to important neurons weights
            if i in important_neurons:
                mask = np.zeros_like(weights)  
                important_indices = important_neurons[i]   #important_indices is a list of indices of important nodes
                mask[..., important_indices] = 1           # mask=1 for the important neurons
                weights = weights * mask   

                #now remove the biases from unimportant nodes i.e. take biases only from important nodes
                new_biases = np.zeros_like(biases)
                new_biases[important_indices] = biases[important_indices]
                biases = new_biases
            
            # Add a new Conv2D layer with the modified weights and biases
            sub_network_layers.append(layers.Conv2D(filters=layer.filters, 
                                                     kernel_size=layer.kernel_size, 
                                                     strides=layer.strides, 
                                                     padding=layer.padding,
                                                     activation=layer.activation,
                                                     kernel_initializer=tf.constant_initializer(weights),
                                                     bias_initializer=tf.constant_initializer(biases)))
        
        elif isinstance(layer, layers.Dense):
            weights, biases = layer.get_weights()
            
            if i in important_neurons:
                mask = np.zeros_like(weights)
                important_indices = important_neurons[i]
                mask[:, important_indices] = 1   
                weights = weights * mask

                new_biases = np.zeros_like(biases)
                new_biases[important_indices] = biases[important_indices]
                biases = new_biases
            
            sub_network_layers.append(layers.Dense(
                units=layer.units, 
                activation=layer.activation,
                kernel_initializer=tf.constant_initializer(weights),
                bias_initializer=tf.constant_initializer(biases)
            ))
        
        elif isinstance(layer, layers.MaxPooling2D):
            sub_network_layers.append(layer)
        
        elif isinstance(layer, layers.Flatten):
            sub_network_layers.append(layer)
        
        else:
            # Any other layers are added as they are
            sub_network_layers.append(layer)

    sub_network = tf.keras.Sequential(sub_network_layers)
    
    sub_network.build(input_shape=original_model.input_shape)
    sub_network.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])

    return sub_network


def get_important_neurons(model, x_test, y_test, which_class, thresh):
    '''
    params:
    model: original model from which we subselect important neurons
    x_test, y_test= unseen test dataset that are used for calculating activations
    which_class= for which class we're identifying important neurons?
    thresh= important neurons belong above what percentile?

    returns:
    important_neurons: a dict --> {layer_number: [indices of imp neurons of that layer for class= which_class]}
    unimportant_neurons: similar to the above, this is required if we want to do the ablation study
    '''
    class_indices = np.where(np.argmax(y_test, axis=1) == which_class)[0]
    print(f"Class {which_class} has {len(class_indices)} samples")
    x_input = x_test[class_indices]

    # Define input explicitly for activation model
    input_layer = tf.keras.Input(shape=x_input.shape[1:])  # Shape: (28, 28, 1)
    layer_outputs = [input_layer]
    for layer in model.layers:
        layer_outputs.append(layer(layer_outputs[-1]))
    
    activation_model = tf.keras.Model(inputs=input_layer, outputs=layer_outputs[1:])  # Skip input layer

    activations = activation_model.predict(x_input)

    important_neurons = {}
    unimportant_neurons= {}
    important_neurons_activations = {}
    for i, (layer, layer_act) in enumerate(zip(model.layers, activations)):
        # Skip layers that are not Conv2D or Dense
        if not isinstance(layer, (layers.Conv2D, layers.Dense)):
            continue

        if len(layer_act.shape) == 4:
            avg_activation = np.mean(layer_act, axis=(0, 1, 2))
        else:
            avg_activation = np.mean(layer_act, axis=0)
        
        #take all neurons from the last layer
        if (i == len(model.layers)-1):
           important_neurons[i]= [0,1,2,3,4,5,6,7,8,9]
           continue

        threshold = np.percentile(avg_activation, thresh)
        # Get important neurons (indices where activation > threshold)
        neuron_indices = np.where(avg_activation > threshold)[0]
        unimp_neuron_indices= np.where(avg_activation < threshold)[0]
        unimportant_neurons[i]= unimp_neuron_indices
        important_neurons[i] = neuron_indices
        # Store activation values for important neurons in this layer
        layer_activations = {}
        for neuron in neuron_indices:
            layer_activations[neuron] = avg_activation[neuron]
        important_neurons_activations[i] = layer_activations
    return important_neurons, important_neurons_activations, unimportant_neurons


def main(args):
   
    dataset= args.dataset
    x_train, y_train, x_test, y_test= dataloader.get_preprocessed_data(dataset)     #ndarrays
    original_model= load_model(f"trained_models/{dataset}/original_model.h5")

    #uncomment the line below to do P_reuse and P_specialize experiments
    P_spec_and_P_reuse(original_model, x_test, y_test, dataset, 1)
    exit()
    results = {}
    class_neurons = {}
    all_unique_neurons = set()
    subnetwork_graphs = []

    y_pred= original_model.predict(x_test)
    orig_pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    for which_class in range(10):
        
        print(f"Class: {which_class}")
        imp_neurons, imp_activations, unimp_neurons = get_important_neurons( original_model, x_test, y_test, which_class, thresh=70)

        class_neurons[which_class] = imp_neurons
        
        masked_model= create_sub_network( original_model, imp_neurons)
        
        # Build graph from masked model (for one digit, or iterate for each digit)
        G_class = vis.assign_positions(vis.build_graph_all_layers(masked_model, imp_neurons, layer_activation_stats=imp_activations))
        # Append the tuple (class_label, graph) for later merging.
        subnetwork_graphs.append((which_class, G_class))        
        
        class_indices= np.where(np.argmax(y_test, axis=1) == which_class)[0]
        x_input = x_test[class_indices]
        y_input= y_test[class_indices]
        
        _, acc_ownClass = masked_model.evaluate(x_input, y_input)

        # Evaluate on all digits
        _, full_acc = masked_model.evaluate(x_test,y_test) 

        #get predictions of the masked model
        predicted_probs_masked_model= masked_model.predict(x_test)
        pred_labels_mm= np.argmax(predicted_probs_masked_model, axis=1)
        unique_preds= np.unique(pred_labels_mm)
        print(f'Masked model for class {which_class} maps everything in classes: {unique_preds}')
        #Modularity Score: How much accuracy is retained on correct vs. all digits
        modularity_score = acc_ownClass / full_acc
        #print(f"Modularity Score: {modularity_score}") 
        
        results[which_class] = {
            "accuracy on own class": acc_ownClass,
            "accuracy on entire dataset": full_acc,
            "modularity_score": modularity_score
        }
        
        activation_stats = vis.get_activation_stats(imp_activations)
        #imp_neurons_per_class = activation_stats['total_neurons']
        print(activation_stats)
        all_unique_neurons.update({neuron for layer_neurons in imp_neurons.values() for neuron in layer_neurons}) 

        #ablation study-- not relevant for this experiment since it shows little specialization
        # ablated_model= create_sub_network( original_model, unimp_neurons)
        # abl_preds = ablated_model.predict(x_test, verbose=0)
        # abl_pred_classes = np.argmax(abl_preds, axis=1)

        # # Get the confusion matrix
        # cm_original = confusion_matrix(true_classes, orig_pred_classes, labels=np.arange(10))
        # cm_ablated = confusion_matrix(true_classes, abl_pred_classes, labels=np.arange(10))
        # cm_diff=  cm_ablated - cm_original 
        # vis.visualize_cm_diff(cm_diff, dataset, which_class)

    # Combine all subnetwork graphs into one
    G_all = vis.build_combined_graph(subnetwork_graphs)

    # Export the combined graph as a GEXF file
    nx.write_gexf(G_all, "all_subnetworks.gexf")
    [print(f"Class {i}: {results[i]}") for i in range(10)]
    

if __name__ == "__main__":
    
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'mnist')
    args = parser.parse_args()

    assert args.dataset in ['mnist', 'cifar10']
    main(args)