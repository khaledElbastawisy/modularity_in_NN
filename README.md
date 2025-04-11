## Instructions to run

Run src/main.py. It will--
* load a pretrained CNN model (trained on MNIST data)
* for a particular class, get a dictionary of important neurons of the CNN model ($which_class$ in main method defines the class for which we are identifying the important neurons)
* create a sub-network with only the important nodes
* test the sub network on 2 types of datasets: (a) whole test dataset and (b) only the class for which we identified the important nodes ($which_class$) and report the accuracy
