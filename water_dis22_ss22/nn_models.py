from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D
import inspect
import tensorflow as tf
from keras.applications.vgg16 import VGG16



def add_classification_top_layer(model, out_classes, neurons_l, dropout=0.5, unfreeze_layers=0):
    """Adds custom top layers

    Args:
        model (keras model): Pass the base model
        out_classes (int): Define how many classes the classification layer shall have
        neurons_l (list): Define how many hidden layers the top model shall have (len(neurons_l)) and how many neurons
            these layers shall have (ints)
        dropout (float): Define the dropout in these hidden layers
        unfreeze_layers (int): Percentage of how many layers shall be learnable/unfrozen

    Returns:
        model (keras model): Final model
    """
    if unfreeze_layers:
        #freeze layers of input model
        #unfreeze at least one layer if unfreeze layers != 0
        unfrozen_layers = max(1, round(len(model.layers) * unfreeze_layers/100))
        freeze_layers = len(model.layers) - unfrozen_layers
        for layer in model.layers[0:freeze_layers]:
            layer.trainable = False
        print('Frozen layers', freeze_layers, 'unfrozen layers', unfrozen_layers, 'ges_layers', len(model.layers))
    #Add extra layers and always pass the output tensor to next layer
    x = model.output
    x = GlobalAveragePooling2D()(x)
    #add multiple layers defined in neurons_l
    for neurons in neurons_l:
        x = Dense(neurons, activation='relu')(x)
        if dropout:
            x = Dropout(dropout)(x)
    #add softmax for
    out = Dense(out_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=model.input, outputs=out)
    return model

