# Neuron Analysis for CNNs

This repository contains implementations of neuron thresholding and neuron masking algorithms to analyze important neurons in Convolutional Neural Networks.

## Instructions to Run

### Neuron Thresholding Algorithm
```bash
python src/neuron_thresholding.py [--dataset mnist|cifar10]
```

### Neuron Masking Algorithm
```bash
python src/neuron_masking.py [--dataset mnist|cifar10]
```

By default, both algorithms use the CIFAR10 dataset if no `--dataset` flag is provided.

## What the Algorithms Do

When executed, each algorithm will:

1. Load a pretrained CNN model (trained on the specified dataset)
2. Identify important neurons for a particular class (configurable via `which_class` in the main method)
3. Create a sub-network containing only the important neurons
4. Test the sub-network on:
   - The complete test dataset
   - Only test samples from the class for which important neurons were identified
5. Analyze neuron reuse and specialization for each sub-network
6. Create an ablation model to measure the impact of removing the sub-network
7. Generate and export visualization plots supporting the conclusions in the accompanying report

## Requirements

The repository requires standard deep learning libraries such as PyTorch, along with data visualization tools like Matplotlib.

## Results

The algorithms produce visualizations that help understand:
- Which neurons are most important for classifying specific objects
- How neurons are shared across different classification tasks
- The degree of specialization vs. generalization in neural networks

For detailed analysis and conclusions, please refer to the accompanying report.
