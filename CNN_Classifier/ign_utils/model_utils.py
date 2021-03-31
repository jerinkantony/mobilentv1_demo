import tensorflow as tf
import tensorflow.keras

def limit_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
      except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


def unfreeze_layers(model, numunfreeze=None):
    if numunfreeze is None:
        for layer in model.layers:
            layer.trainable=True
            print(layer, layer.trainable)
    else:   
        for layer in model.layers[:-numunfreeze]:
            print(layer, layer.trainable)
            layer.trainable=False
        for layer in model.layers[-numunfreeze:]:
            layer.trainable=True
            print(layer, layer.trainable)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def copy_weights(curr_model, transfer_weights_path='model.h5', till_dense=True):
    #import pdb;pdb.set_trace()
    loaded_model = keras.models.load_model(transfer_weights_path)
    #loaded_model.summary()
    for i, layer in enumerate(loaded_model.layers):
        last = layer.name
        if ('dense' in layer.name) and (till_dense==True): 
            break
        try:
            curr_model.layers[i].set_weights(loaded_model.layers[i].get_weights())
            curr_model.layers[i].trainable = False
            print('copied ',layer.name)
        except:
            print( 'not able to copy',layer.name) 
            #break
    print('Copied weights till',last, '(Including',last,')') 
    return curr_model
        
   
def copy_weightsold(target_model, source_weights='model.h5', till_dense=False):

    loaded_model = keras.models.load_model(source_weights)
    #loaded_model.summary()
    for i, layer in enumerate(loaded_model.layers):
        print('copying ',layer.name)
        last = layer.name
        if ('dense' in layer.name) and (till_dense==True): 
            break
        try:
            target_model.layers[i].set_weights(loaded_model.layers[i].get_weights())
        except:
            print( 'not able to copy ',last, 'breaking') 
            break
    print('Copied weights till',last, '(Including',last,')') 
    #import pdb;pdb.set_trace()
          
    print('transfer_weights changed to null in json')
#copy_weights(target_model, source_weight_name = 'test.h5', till_dense=False) 

def copy_cudnn_to_noncudnn(cudnn_lstm_model, lstm_model):
    from keras.engine.saving import preprocess_weights_for_loading

    cudnn_weights = cudnn_lstm_model.get_weights()
    weights2 = preprocess_weights_for_loading(lstm_layer, cudnn_weights) # target layer, source weights
    lstm_model.set_weights(cudnn_weights)




  
  
