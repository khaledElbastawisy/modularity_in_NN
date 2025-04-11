

input_shape = None 
batch_size= None 
num_classes= None 
epochs= None 

def set_vars(dataset):
    global input_shape, batch_size, num_classes, epochs 
    if (dataset == 'mnist'):
        input_shape = (28, 28, 1)
        batch_size = 64
        num_classes = 10
        epochs = 5