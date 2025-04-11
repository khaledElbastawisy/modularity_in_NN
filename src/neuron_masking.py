import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import argparse 
import networkx as nx
import random
from sklearn.metrics import confusion_matrix
from scipy.stats import norm  # For Bayesian posterior probability computation
import visualization_neuron_masking as vis

#############################
# Strategy 4: Temperature Scaling Function
#############################
def calibrate_predictions(predictions, temperature=2.0):
    """
    Applies temperature scaling to soften the predictions.
    predictions: numpy array of softmax outputs.
    temperature: positive scalar > 0. A higher temperature produces softer probabilities.
    """
    logits = np.log(predictions + 1e-8)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits)
    calibrated = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return calibrated

#############################
# Subnetwork Creation Functions
#############################
def create_sub_network(original_model, neuron_masks):
    """
    Creates a compressed subnetwork using the neuron masks.
    neuron_masks: dictionary mapping layer index -> mask array.
      Mask values:
         1  => force neuron output to 1 (by zeroing its weights and setting bias to 1),
         0  => force neuron output to 0 (by zeroing its weights and setting bias to 0),
        -1  => leave neuron unchanged.
    """
    sub_network_layers = []
    for i, layer in enumerate(original_model.layers):
        if isinstance(layer, layers.Conv2D):
            weights, biases = layer.get_weights()
            if i in neuron_masks:
                mask = neuron_masks[i]
                for j in range(mask.shape[0]):
                    if mask[j] == 1:
                        weights[..., j] = 0
                        biases[j] = 1
                    elif mask[j] == 0:
                        weights[..., j] = 0
                        biases[j] = 0
            new_conv = layers.Conv2D(filters=layer.filters,
                                     kernel_size=layer.kernel_size,
                                     strides=layer.strides,
                                     padding=layer.padding,
                                     activation=layer.activation,
                                     kernel_initializer=tf.constant_initializer(weights),
                                     bias_initializer=tf.constant_initializer(biases))
            sub_network_layers.append(new_conv)

        elif isinstance(layer, layers.Dense):
            weights, biases = layer.get_weights()
            if i in neuron_masks:
                mask = neuron_masks[i]
                for j in range(mask.shape[0]):
                    if mask[j] == 1:
                        weights[:, j] = 0
                        biases[j] = 1
                    elif mask[j] == 0:
                        weights[:, j] = 0
                        biases[j] = 0
            new_dense = layers.Dense(units=layer.units,
                                     activation=layer.activation,
                                     kernel_initializer=tf.constant_initializer(weights),
                                     bias_initializer=tf.constant_initializer(biases))
            sub_network_layers.append(new_dense)
        
        elif isinstance(layer, layers.MaxPooling2D):
            sub_network_layers.append(layer)
        elif isinstance(layer, layers.Flatten):
            sub_network_layers.append(layer)
        else:
            sub_network_layers.append(layer)

    sub_network = tf.keras.Sequential(sub_network_layers)
    sub_network.build(input_shape=original_model.input_shape)
    sub_network.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08),
                        loss='categorical_crossentropy', metrics=['acc'])
    return sub_network

#############################
# Selective Bayesian Masking Function (Modified)
#############################
def get_neuron_masks(model, x_test, y_test, which_class, t_high=0.5, t_low=1e-6, confidence=0.8,
                     base_fraction=0.1, max_fraction=0.3):
    """
    Computes a mask for each neuron in weighted layers (Conv2D or Dense) using a Bayesian decision-theoretic approach.
    
    For each neuron:
      - Compute the sample mean and standard error of its activations (over all samples for the class).
      - Compute the probability (using a normal approximation) that the neuron's true mean > t_high and < t_low.
      - Instead of using a fixed fraction for all layers, we compute a dynamic fraction for the current weighted layer.
        The fraction is set to be higher for later layers (e.g., from base_fraction up to max_fraction).
      - For a given layer, only the top fraction (as computed) of neurons (by extremeness of p_high for "on" and p_low for "off")
        are fixed if the corresponding probability exceeds the confidence threshold.
      - Otherwise, the neuron's mask is left as -1 (allowing the network to "read" the neuron).
    
    Returns:
      neuron_masks: dictionary mapping layer index -> mask array.
      neuron_stats: dictionary mapping layer index -> statistics (mean, std, se, p_high, p_low) for each neuron.
    """
    # Identify test samples of the target class.
    class_indices = np.where(np.argmax(y_test, axis=1) == which_class)[0]
    print(f"Class {which_class} has {len(class_indices)} samples")
    x_input = x_test[class_indices]

    # Build an activation model that outputs each layer's activations.
    input_layer = tf.keras.Input(shape=x_input.shape[1:])
    layer_outputs = [input_layer]
    for layer in model.layers:
        layer_outputs.append(layer(layer_outputs[-1]))
    activation_model = tf.keras.Model(inputs=input_layer, outputs=layer_outputs[1:])
    
    activations = activation_model.predict(x_input)
    
    # First, collect indices of weighted layers.
    weighted_layers = []
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, (layers.Conv2D, layers.Dense)):
            weighted_layers.append(idx)
    num_weighted = len(weighted_layers)
    
    neuron_masks = {}
    neuron_stats = {}
    
    # Iterate over layers along with their index and activation.
    for i, (layer, layer_act) in enumerate(zip(model.layers, activations)):
        #take all neurons from last layer
        if (i == len(model.layers)-1):
           neuron_masks[i]= np.full((10),-1)
           continue
        if not isinstance(layer, (layers.Conv2D, layers.Dense)):
            continue

        # Compute a dynamic fraction for this layer based on its order among weighted layers.
        order = weighted_layers.index(i)
        # Linear interpolation between base_fraction and max_fraction:
        if num_weighted > 1:
            layer_fraction = base_fraction + (order / (num_weighted - 1)) * (max_fraction - base_fraction)
        else:
            layer_fraction = base_fraction

        # For Conv2D: average over spatial dimensions.
        if len(layer_act.shape) == 4:
            reshaped = np.mean(layer_act, axis=(1, 2))
        else:
            reshaped = layer_act
        
        n_samples = reshaped.shape[0]
        mu = np.mean(reshaped, axis=0)
        sigma = np.std(reshaped, axis=0) + 1e-8  # Avoid zero division
        se = sigma / np.sqrt(n_samples)
        
        # Compute posterior probabilities using the normal approximation.
        p_high = 1 - norm.cdf((t_high - mu) / se)
        p_low = norm.cdf((t_low - mu) / se)
        
        mask = np.full(mu.shape, -1)
        n_neurons = len(mu)
        fix_count = int(n_neurons * layer_fraction)
        
        # For "always on": sort by descending p_high and fix only the top ones.
        if fix_count > 0:
            high_sorted_idx = np.argsort(p_high)[::-1]
            high_fixed_indices = high_sorted_idx[:fix_count]
            for idx in high_fixed_indices:
                if p_high[idx] > confidence:
                    mask[idx] = 1
            # For "always off": sort by ascending p_low and fix only the top ones.
            low_sorted_idx = np.argsort(p_low)
            low_fixed_indices = low_sorted_idx[:fix_count]
            for idx in low_fixed_indices:
                if p_low[idx] > confidence:
                    mask[idx] = 0

        neuron_masks[i] = mask
        neuron_stats[i] = {'mean': mu, 'std': sigma, 'se': se, 'p_high': p_high, 'p_low': p_low,
                           'fraction_used': layer_fraction}
    
    return neuron_masks, neuron_stats

#############################
# Ablation Evaluation Functions 
#############################
def create_ablation_model(original_model, all_class_masks, removed_class):
    """
    Creates a model where the subnetwork associated with the specified class is ablated (removed).
    
    Parameters:
        original_model: The original trained model
        all_class_masks: Dictionary mapping class index to its corresponding neuron masks
        removed_class: The class whose associated subnetwork will be removed
        
    Returns:
        A model with the specified class's subnetwork removed
    """
    # Get the mask for the class to be removed
    if removed_class not in all_class_masks:
        raise ValueError(f"No mask data for class {removed_class}")
    
    removed_mask = all_class_masks[removed_class]
    
    # Create an inverted mask that zeros out all neurons that were specifically
    # masked for the removed class
    ablation_masks = {}
    for layer_idx, mask in removed_mask.items():
        inverted_mask = np.full_like(mask, -1)  # Default to unchanged
        
        # For neurons that were fixed (0 or 1) for the removed class,
        # force them to the opposite value or zero them out
        for i in range(len(mask)):
            if mask[i] == 1:  # If neuron was forced to 1, force to 0
                inverted_mask[i] = 0
            elif mask[i] == 0:  # If neuron was forced to 0, leave as is or force to 0
                inverted_mask[i] = 0
        
        ablation_masks[layer_idx] = inverted_mask
    
    # Create the ablated model
    ablated_model = create_sub_network(original_model, ablation_masks)
    return ablated_model

def evaluate_ablation(original_model, all_class_masks, x_test, y_test, removed_class, temperature=2.0):
    """
    Evaluates the effect of removing a class-specific subnetwork on the model's performance.
    
    Parameters:
        original_model: The original trained model
        all_class_masks: Dictionary mapping class index to its corresponding neuron masks
        x_test: Test data
        y_test: Test labels
        removed_class: The class whose associated subnetwork will be removed
        temperature: Parameter for softmax calibration
        
    Returns:
        Dictionary with performance metrics before and after ablation
    """
    # Evaluate original model on all classes
    print(f"\nEvaluating effect of removing subnetwork for class {removed_class}")
    
    # Get predictions from original model
    orig_preds = original_model.predict(x_test, verbose=0)
    orig_calibrated = calibrate_predictions(orig_preds, temperature)
    orig_pred_classes = np.argmax(orig_calibrated, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate class-wise accuracy for original model
    orig_class_acc = {}
    for cls in range(10):  # Assuming 10 classes for CIFAR10
        cls_indices = np.where(true_classes == cls)[0]
        if len(cls_indices) > 0:
            cls_acc = np.mean(orig_pred_classes[cls_indices] == cls)
            orig_class_acc[cls] = cls_acc
    
    # Create the ablated model
    ablated_model = create_ablation_model(original_model, all_class_masks, removed_class)
    
    # Evaluate ablated model on all classes
    abl_preds = ablated_model.predict(x_test, verbose=0)
    abl_calibrated = calibrate_predictions(abl_preds, temperature)
    abl_pred_classes = np.argmax(abl_calibrated, axis=1)

    #AFIA CM
    # Get the confusion matrix
    cm_original = confusion_matrix(true_classes, orig_pred_classes, labels=np.arange(10))
    cm_ablated = confusion_matrix(true_classes, abl_pred_classes, labels=np.arange(10))
    #print(cm_original)
    #print(cm_ablated)
    #exit()
    cm_diff=  cm_ablated - cm_original 
    
    # Calculate class-wise accuracy for ablated model
    abl_class_acc = {}
    for cls in range(10):
        cls_indices = np.where(true_classes == cls)[0]
        if len(cls_indices) > 0:
            cls_acc = np.mean(abl_pred_classes[cls_indices] == cls)
            abl_class_acc[cls] = cls_acc
    
    # Calculate changes in accuracy
    acc_changes = {}
    for cls in range(10):
        if cls in orig_class_acc and cls in abl_class_acc:
            acc_changes[cls] = abl_class_acc[cls] - orig_class_acc[cls]
    
    # Calculate confusion matrix to understand which classes are mistaken for which
    confusion = np.zeros((10, 10))
    for true_cls in range(10):
        cls_indices = np.where(true_classes == true_cls)[0]
        if len(cls_indices) > 0:
            pred_dist = np.zeros(10)
            for i in range(10):
                pred_dist[i] = np.sum(abl_pred_classes[cls_indices] == i) / len(cls_indices)
            confusion[true_cls] = pred_dist
    
    # Focus on the removed class
    removed_class_indices = np.where(true_classes == removed_class)[0]
    if len(removed_class_indices) > 0:
        removed_class_acc_before = orig_class_acc[removed_class]
        removed_class_acc_after = abl_class_acc[removed_class]
        removed_class_effect = removed_class_acc_after - removed_class_acc_before
        
        # Calculate new prediction distribution for the removed class
        removed_class_pred_dist = confusion[removed_class]
    else:
        removed_class_acc_before = 0
        removed_class_acc_after = 0
        removed_class_effect = 0
        removed_class_pred_dist = np.zeros(10)
    
    results = {
        "removed_class": removed_class,
        "original_accuracy": orig_class_acc,
        "ablated_accuracy": abl_class_acc,
        "accuracy_changes": acc_changes,
        "confusion_matrix": cm_diff,#confusion,
        "removed_class_acc_before": removed_class_acc_before,
        "removed_class_acc_after": removed_class_acc_after,
        "removed_class_effect": removed_class_effect,
        "removed_class_pred_dist": removed_class_pred_dist
    }
    
    return results

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
        
        neuron_masks_1, _ = get_neuron_masks(
            original_model, x_class_1, y_class_1, classes[0], t_high=0.55, t_low=1e-5,
            confidence=0.90, base_fraction=0.1, max_fraction=0.3
        )

        neuron_masks_2, _ = get_neuron_masks(
            original_model, x_class_2, y_class_2, classes[1], t_high=0.55, t_low=1e-5,
            confidence=0.90, base_fraction=0.1, max_fraction=0.3
        )       
        
        #neuron_mask_i is a dictionary. {layer_num: array([-1,1,0,....])}
        for layer_num, mask_array_1 in neuron_masks_1.items():
            mask_array_2= neuron_masks_2[layer_num]
            #consider only ACTIVE neurons
            similar_masks = np.where((mask_array_1 == 1) & (mask_array_2 == 1), 1, 0)
            similarity_ratio= np.sum(similar_masks)/len(mask_array_1)
            neuron_share_ratio[layer_num] = neuron_share_ratio[layer_num] + similarity_ratio

    #take average over num_repeat tests
    for i in range (len(original_model.layers)):
        neuron_share_ratio[i]= neuron_share_ratio[i]/num_repeat
    print(neuron_share_ratio)

    #Now plot the results
    vis.visualize_P_results(neuron_share_ratio, datasetName, P_val)


#############################
# Main Routine with Ablation Evaluation
#############################
def main(args):
    dataset = args.dataset
    # Use the dataloader.py file
    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        # Normalize pixel values to be between 0 and 1
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        # Convert labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Expand dimensions to (None, 28, 28, 1)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        # Normalize pixel values to be between 0 and 1
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        # Convert labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    else:
        raise ValueError("Invalid dataset. Choose 'mnist' or 'cifar10'.")
    
    original_model = load_model(f"trained_models/{dataset}/original_model.h5")

    results = {}
    all_unique_neurons = set()
    subnetwork_graphs = []
    all_class_masks = {}  # Store masks for all classes for ablation study

    # Temperature for calibration of the softmax outputs.
    temperature = 2.0

    #uncomment the line below for P_reuse and P_specialize experiments
    #P_spec_and_P_reuse(original_model, x_test, y_test, dataset , P_val=1)
    #exit()
    for which_class in range(10):
        print(f"Class: {which_class}")
        # Compute neuron masks using the modified (selective by layer order) Bayesian approach.
        neuron_masks, neuron_stats = get_neuron_masks(
            original_model, x_test, y_test, which_class, t_high=0.55, t_low=1e-5,
            confidence=0.90, base_fraction=0.1, max_fraction=0.3
        )
        
        # Store masks for ablation study
        all_class_masks[which_class] = neuron_masks
        
        # Create the subnetwork based on the computed masks.
        masked_model = create_sub_network(original_model, neuron_masks)
        
        # (Optional) Fine-tune the masked model here if desired.
        # e.g., masked_model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.1)
        
        # Build the graph for the subnetwork.
        G_class = vis.assign_positions(vis.build_graph_all_layers(masked_model, neuron_masks))
        subnetwork_graphs.append((which_class, G_class))
        
        # Evaluate on class-specific samples.
        class_indices = np.where(np.argmax(y_test, axis=1) == which_class)[0]
        x_input = x_test[class_indices]
        y_input = y_test[class_indices]
        _, acc_ownClass = masked_model.evaluate(x_input, y_input, verbose=0)
        
        # Evaluate on the entire test set.
        _, full_acc = masked_model.evaluate(x_test, y_test, verbose=0)

        #print which class it maps everything to?
        predicted_probs_masked_model= masked_model.predict(x_test)
        pred_labels_mm= np.argmax(predicted_probs_masked_model, axis=1)
        unique_preds= np.unique(pred_labels_mm)
        #print(f'Masked model for class {which_class} maps everything in classes: {unique_preds}')
        
        # Get predictions and apply temperature scaling.
        predictions = masked_model.predict(x_test, verbose=0)
        calibrated_preds = calibrate_predictions(predictions, temperature=temperature)
        y_pred = np.argmax(calibrated_preds, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Compute precision for the subnetwork as a binary classifier for class 'which_class'.
        tp = np.sum((y_true == which_class) & (y_pred == which_class))
        fp = np.sum((y_true != which_class) & (y_pred == which_class))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        modularity_score = acc_ownClass * precision
        
        results[which_class] = {
            "accuracy on own class": acc_ownClass,
            "accuracy on entire dataset": full_acc,
            "precision": precision,
            "modularity_score": modularity_score
        }
        
        #stats = get_activation_stats(neuron_stats)
        #print(f"Activation stats for class {which_class}: {stats}")
        for layer_idx, mask in neuron_masks.items():
            all_unique_neurons.update({f"{layer_idx}_{i}" for i in range(len(mask)) if mask[i] != -1})
    
    G_all = vis.build_combined_graph(subnetwork_graphs)
    nx.write_gexf(G_all, "all_subnetworks.gexf")
    
    # Print results for each class
    for i in range(10):
        print(f"Class {i}: {results[i]}")
    print(f"Total unique neurons fixed by decision rule: {len(all_unique_neurons)}")
    
    # Perform ablation study
    print("\n======= ABLATION STUDY =======")
    ablation_results = {}
    
    # For each class, create a model with that class's subnetwork removed and evaluate
    for removed_class in range(10):
        ablation_result = evaluate_ablation(
            original_model, all_class_masks, x_test, y_test, removed_class, temperature
        )
        ablation_results[removed_class] = ablation_result
        
        # Print summary of results for this ablation
        print(f"\nAblation results for removing class {removed_class}:")
        print(f"  Original accuracy for class {removed_class}: {ablation_result['removed_class_acc_before']:.4f}")
        print(f"  Accuracy after ablation: {ablation_result['removed_class_acc_after']:.4f}")
        print(f"  Change: {ablation_result['removed_class_effect']:.4f}")
        
        # Print effect on other classes
        print("  Effect on other classes:")
        for cls in range(10):
            if cls != removed_class:
                change = ablation_result['accuracy_changes'].get(cls, 0)
                print(f"    Class {cls}: {change:.4f}")
    
    # Visualize the ablation results
    vis.visualize_ablation_results(ablation_results)
    print("\nAblation study complete. Visualization saved to 'ablation_results.png'")

    # Visualize all confusion matrices in a separate image
    vis.visualize_cm_diff(ablation_results, dataset=args.dataset)
    print("\nAblation study complete. Visualization saved to 'ablation_results.png'")
    print(f"Confusion matrices visualization saved to 'class_ablation_confusion_matrices.png'")



if __name__ == "__main__":
    np.random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cifar10')  # changed default to cifar10
    args = parser.parse_args()
    assert args.dataset in ['mnist', 'cifar10']  # added cifar10 to allowed datasets
    main(args)
