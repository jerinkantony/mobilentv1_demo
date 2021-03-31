import os

def check():
    print('hello from ml_utils')
    
def getmodel(model_nam,HEIGHT, WIDTH, CHANNEL=3):
    '''Returns pretrained base model and preprocessing_function '''
    
    preprocessing_function = None
    base_model = None
    
    if CHANNEL!=3:
        assert('CHANNEL!=3, pretrained models are three channel. ')
    # Prepare the model
    if model_nam == "VGG16":
        from keras.applications.vgg16 import VGG16,preprocess_input
        preprocessing_function = preprocess_input
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    elif model_nam == "VGG19":
        from keras.applications.vgg19 import VGG19, preprocess_input
        preprocessing_function = preprocess_input
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    elif model_nam == "ResNet50":
        from keras.applications.resnet50 import ResNet50, preprocess_input
        preprocessing_function = preprocess_input
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    elif model_nam == "InceptionV3":
        from keras.applications.inception_v3 import InceptionV3, preprocess_input
        preprocessing_function = preprocess_input
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    elif model_nam == "Xception":
        from keras.applications.xception import Xception, preprocess_input
        preprocessing_function = preprocess_input
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    elif model_nam == "InceptionResNetV2":
        from keras.applications.inceptionresnetv2 import InceptionResNetV2, preprocess_input
        preprocessing_function = preprocess_input
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    elif model_nam == "MobileNet":
        from keras.applications.mobilenet import MobileNet,preprocess_input
        preprocessing_function = preprocess_input
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    elif model_nam == "DenseNet121":
        from keras.applications.densenet import DenseNet121, preprocess_input
        preprocessing_function = preprocess_input
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    elif model_nam == "DenseNet169":
        from keras.applications.densenet import DenseNet169,preprocess_input
        preprocessing_function = preprocess_input
        base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    elif model_nam == "DenseNet201":
        from keras.applications.densenet import DenseNet201, preprocess_input
        preprocessing_function = preprocess_input
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    elif model_nam == "NASNetLarge":
        from keras.applications.nasnet import NASNetLarge, preprocess_input
        preprocessing_function = preprocess_input
        base_model = NASNetLarge(weights='imagenet', include_top=True, input_shape=(HEIGHT, WIDTH, 3))
    elif model_nam == "NASNetMobile":
        from keras.applications.nasnet import NASNetMobile, preprocess_input
        preprocessing_function = preprocess_input
        base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    else:
        ValueError("The model you requested is not supported in Keras")
        
    base_model.summary()
    return base_model, preprocessing_function
    
# For boolean input from the command line
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    


def save_class_list(path, class_list):
    class_list.sort()
    target=open(path + "/class_list.txt",'w')
    for c in class_list:
        target.write(c)
        target.write("\n")

def load_class_list(class_list_file):
    import csv
    class_list = []
    with open(class_list_file, 'r') as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            class_list.append(row)
    class_list.sort()
    return class_list

# Get a list of subfolders in the directory
def get_subfolders(directory):
    subfolders = os.listdir(directory)
    subfolders.sort()
    return subfolders

# Get number of files by searching directory recursively
def get_num_files(directory):
    import glob
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

# Add on new FC layers with dropout for fine tuning
def build_finetune_model(base_model, dropout, fc_layers, num_classes, freeze=False):
    from keras.layers import Dense, Activation, Flatten, Dropout
    from keras.models import Sequential, Model
    
    for layer in base_model.layers:
        layer.trainable = not(freeze)

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x) # New FC layer, random init
        x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation='softmax')(x) # New softmax layer
    
    base_dense_model = Model(inputs=base_model.input, outputs=predictions)

    return base_dense_model

# Plot the training and validation loss + accuracy
def plot_training(history):
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    #plt.savefig('acc_vs_epochs.png')
    
    
    
if __name__ == '__main__':
    pass
    
    
    
    
    
