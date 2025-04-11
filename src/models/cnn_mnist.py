#run this file to train and save the original model

import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse 
from models import var 
import numpy as np 
import dataloader


def get_compiled_model(dataset, input_shape, num_classes):
    
    if (dataset == 'mnist'):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=input_shape),
            tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=(2,2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])

    return model 

def main(args):
    dataset= args.dataset
    train_model= args.train_model
    var.set_vars(dataset)
    input_shape= var.input_shape
    batch_size = var.batch_size
    num_classes = var.num_classes
    epochs = var.epochs

    x_train, y_train, x_test, y_test= dataloader.get_preprocessed_data(dataset)
    
    
    if (train_model):
        model= get_compiled_model(dataset, input_shape, num_classes)
        
        history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1)
        
        #save model
        model.save(f"../trained_models/{dataset}/original_model.h5")


    model= load_model(f"../trained_models/{dataset}/original_model.h5")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Original model test accuracy: {test_acc}')
    

if __name__ == "__main__":
    
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'mnist')
    parser.add_argument("--train_model", action="store_true")
    args = parser.parse_args()

    assert args.dataset in ['mnist']
    main(args)