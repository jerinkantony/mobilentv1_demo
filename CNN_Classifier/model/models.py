import os
import sys
import json
import os.path as osp
import numpy as np

path = osp.dirname(osp.abspath(__file__))
sys.path.append(path)
sys.path.append(os.path.join(path,'../ign_utils'))

from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Reshape, MaxPooling2D, MaxPool2D, Conv2D,SpatialDropout2D, Dropout, Flatten, Dense,\
    GlobalAveragePooling2D, GlobalMaxPooling2D,GlobalMaxPooling1D, GlobalAvgPool1D, BatchNormalization, Lambda, Activation, DepthwiseConv2D,SeparableConv2D,ZeroPadding2D

from keras.layers import Add, concatenate, ActivityRegularization, GaussianDropout



from keras.preprocessing import image
from general_utils import eucl_dist_output_shape
from model_utils import euclidean_distance
from tensorflow.keras.utils import plot_model


class MODEL():
    def __init__(self, inp_dim=(224, 224, 3), modelname=None, freeze=False, num_unfreeze=5, load_prev_weights=True, 
                            dropout=None, fc_layers=None, num_outnodes=None, mode='regression',pooling_layer='GlobalAvgPool2D'):
        
        self.path                 = osp.dirname(osp.abspath(__file__))
        self.attire_model_dir     = osp.join(self.path, '..', 'weights', 'attire_siamese')
        self.attire_modeljson     = 'attire_model.json'
        self.modelname            = modelname
        self.freeze               = freeze
        self.num_unfreeze         = num_unfreeze
        self.load_prev_weights    = load_prev_weights
        self.seq_length           = inp_dim[0]
        self.dropout              = dropout
        self.fc_layers            = fc_layers
        self.num_outnodes         = num_outnodes
        self.mode                 = mode
        self.weights              = None
        self.base_model           = None
        self.inp_dim              = None
        
        if pooling_layer   == 'GlobalAvgPool':
            if len(inp_dim) == 2:
                self.pooling_layer = GlobalAvgPool1D()
            else:
                self.pooling_layer = GlobalAveragePooling2D()
        elif pooling_layer == 'GlobalMaxPool':
            if len(inp_dim) == 2:
                self.pooling_layer = GlobalMaxPooling1D()
            else:
                self.pooling_layer = GlobalMaxPooling2D()
        elif pooling_layer == 'Flatten':
            self.pooling_layer = Flatten()
        else:
            self.pooling_layer = None


        if load_prev_weights is False:
            self.weights = 'imagenet'


        if self.mode       == 'binary':
            self.activation_fun = 'sigmoid'
        elif self.mode     == 'categorical':
            self.activation_fun = 'softmax'
        elif self.mode     == 'regression':
            self.activation_fun = None #None #'linear'
        else:
            print('\nSelected "{}" is invalid mode..!\n'.format(self.mode))
            sys.exit()
        if len(inp_dim)==3:
            self.HEIGHT, self.WIDTH, self.num_channels = inp_dim[-3:]
        else:
            self.inp_dim = inp_dim
    
    def get_base_model(self):
        self.preprocessing_function=None
        if self.modelname == "attire_model":
            from keras.applications.mobilenet import preprocess_input
            attire_modeljson = osp.join(self.attire_model_dir, self.attire_modeljson)
            attire_modelfile = osp.join(self.attire_model_dir, self.attire_modeljson.replace('.json', '.h5'))
            if osp.isfile(attire_modeljson):
                self.base_model = self.load_model_from_json(attire_modeljson)
                if self.load_prev_weights is False:
                    self.base_model.load_weights(attire_modelfile)
            else:
                print('Unable to get Base model')

        elif self.modelname == "VGG16":
            from keras.applications.vgg16 import VGG16, preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = VGG16(weights=self.weights, include_top=False, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
        
        elif self.modelname == "VGG19":
            from keras.applications.vgg19 import VGG19, preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = VGG19(weights=self.weights, include_top=False, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
        
        elif self.modelname == "ResNet50":
            from keras.applications.resnet50 import ResNet50, preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = ResNet50(weights=self.weights, include_top=False, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
        
        elif self.modelname == "ResNet18":
            from keras.applications.resnet50 import preprocess_input
            from .resnets.resnet20 import resnet_v1
            self.preprocessing_function = preprocess_input
            self.base_model = resnet_v1( input_shape=(self.HEIGHT, self.WIDTH, self.num_channels), depth=20)
        
        elif self.modelname == "InceptionV3":
            from keras.applications.inception_v3 import InceptionV3, preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = InceptionV3(weights=self.weights, include_top=False, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
        
        elif self.modelname == "Xception":
            from keras.applications.xception import Xception, preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = Xception(weights=self.weights, include_top=False, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
        
        elif self.modelname == "InceptionResNetV2":
            from keras.applications.inceptionresnetv2 import InceptionResNetV2, preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = InceptionResNetV2(weights=self.weights, include_top=False, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
        
        elif self.modelname == "MobileNet":
            from keras.applications.mobilenet import MobileNet, preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = MobileNet(weights=self.weights, include_top=False, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))

        elif self.modelname == "MobileNetv2":
            from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = MobileNetV2(weights=self.weights, include_top=False, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
        
        elif self.modelname == "DenseNet121":
            from keras.applications.densenet import DenseNet121, preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = DenseNet121(weights=self.weights, include_top=False, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
        elif self.modelname == "DenseNet169":
            from keras.applications.densenet import DenseNet169, preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = DenseNet169(weights=self.weights, include_top=False, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
        
        elif self.modelname == "DenseNet201":
            from keras.applications.densenet import DenseNet201, preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = DenseNet201(weights=self.weights, include_top=False, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
        
        elif self.modelname == "NASNetLarge":
            from keras.applications.nasnet import NASNetLarge, preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = NASNetLarge(weights=self.weights, include_top=True, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
        
        elif self.modelname == "NASNetMobile":
            from keras.applications.nasnet import preprocess_input
            self.preprocessing_function = preprocess_input
            self.base_model = NASNetMobile(weights=self.weights, include_top=False, input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
        
        elif self.modelname == "customcnn":
            self.base_model = self.customcnn(input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
            self.preprocessing_function=None
        
        elif self.modelname=="customcnn_grex_plate_classifier":
            self.base_model = self.customcnn_grex_plate_classifier(input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
            self.preprocessing_function=None
        
        elif self.modelname == "RX_model_digit":
            self.base_model = self.RX_model_digit(input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
            self.preprocessing_function=None
        
        elif self.modelname == "People_model_lite":
            self.base_model = self.People_model_lite(input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
            self.preprocessing_function=None
            
        elif self.modelname == "custom_people_model_lite":
            self.base_model = self.custom_people_model_lite(input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
            self.preprocessing_function=None

        elif self.modelname == "skip_smaller_model":
            #from skipmodel import smaller_model
            self.base_model = self.smaller_model(input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
            self.preprocessing_function=None

        elif self.modelname == "RX_model_gear":
            self.base_model = self.RX_model_gear(input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
            self.preprocessing_function=None
        
        elif self.modelname == "regressioncnn":
            self.base_model = self.regressioncnn(input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
            self.preprocessing_function=None
        
        elif self.modelname == "customLSTM":
            from keras.applications.mobilenet import preprocess_input
            self.base_model = MobileNet(weights='imagenet', include_top=False)
            self.preprocessing_function=preprocess_input
        
        elif self.modelname == "Siamese2D":
            self.base_model = self.siamese_base_model_2D(input_shape=(self.HEIGHT, self.WIDTH, self.num_channels))
            self.preprocessing_function=None

        elif self.modelname == 'Conv1DModel':
            from audio_models import Conv1DModel
            self.base_model = Conv1DModel(input_shape=self.inp_dim[:2])
            self.preprocessing_function=None

        elif self.modelname == 'Conv1DGRUModel':
            from audio_models import Conv1DGRUModel
            self.base_model = Conv1DGRUModel(input_shape=self.inp_dim[:2])
            self.preprocessing_function=None
        
        elif self.modelname == 'Conv1DModel_Audio':
            from audio_models import Conv1DModel_Audio
            self.base_model = Conv1DModel_Audio(input_shape=self.inp_dim[:2])
            self.preprocessing_function=None

        elif self.modelname == 'Siamese1D':
            from audio_models import Siamese1D
            self.base_model = Siamese1D(input_shape=self.inp_dim[:2])
            self.preprocessing_function=None

        else:
            print('Invalid/Incompatible model selection..! {} '.format(self.modelname))
            sys.exit()
        return self.base_model

    def load_model_from_json(self, jsonpath):
        with open(jsonpath, 'r+') as json_file:
            architecture = json.load(json_file)
        print('Loading Attire Model from Json File..!')
        model = model_from_json(architecture)
        return model

    def get_fc_model(self, base_model):
        #import pdb;pdb.set_trace()
        if self.modelname=="Siamese2D": #TODO remove this later and bring to main flow
            input_shape=(self.HEIGHT, self.WIDTH, self.num_channels)
            return self.get_siamese_model(input_shape, base_model)
        if self.modelname == 'Siamese1D':
            input_shape = self.inp_dim[:2]
            return self.get_siamese_model(input_shape, base_model)
        if self.fc_layers is not None:
            x = base_model.output
            if self.pooling_layer:
                x = self.pooling_layer(x)
            if self.modelname=='customLSTM':
                from keras.layers.recurrent import LSTM
                from keras.layers.wrappers import TimeDistributed
                inp_layer = Input(shape=(self.seq_length, self.HEIGHT, self.WIDTH, 3))
                feature_model = Model(inputs=base_model.input, outputs=x)
                if self.freeze:
                    feature_model.trainable=False
                encoded_frames = TimeDistributed(feature_model)(inp_layer)
                x = LSTM(128, return_sequences=False, dropout=0.1)(encoded_frames)
            else:
                inp_layer = base_model.input

            for ind, fc in enumerate(self.fc_layers):
                x = Dense(fc, activation='relu')(x) # New FC layer, random init
                x = Dropout(self.dropout[ind])(x)
           
            predictions = Dense(self.num_outnodes, activation=self.activation_fun)(x) # New softmax layer
            model = Model(inputs=inp_layer, outputs=predictions)
            print(model.summary())
            return model
        
        else:
            x = base_model.output
            if self.pooling_layer:
                x = self.pooling_layer(x)
            predictions = Dense(self.num_outnodes, activation=self.activation_fun)(x) # New softmax layer
            inp_layer = base_model.input
            model = Model(inputs=inp_layer, outputs=predictions)
            return model

    def get_model(self):
        base_model = self.get_base_model()
        
        model = self.get_fc_model(base_model)
        return model, self.preprocessing_function

    def siamese_base_model_2D(self, input_shape):
        input = Input(shape=input_shape)
        x = Conv2D(16, (3, 3), padding="valid")(input)
        x = Dropout(0.1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(16, (3, 3), padding="valid")(x)
        x = Dropout(0.2)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(16, (3, 3), padding="valid")(x)
        x = Dropout(0.12)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        return Model(input, x)

    
    def get_siamese_model(self, input_shape, base_model):
        print('Inside Siamese..!')
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        processed_a = base_model(input_a)
        processed_b = base_model(input_b)
        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        model = Model([input_a, input_b], distance)
        print(model.summary())
        return model

    def customcnn(self, input_shape):
        HEIGHT, WIDTH, num_channels = input_shape
        model = Sequential()
        model.add(Conv2D(8, (3, 3), input_shape=(HEIGHT, WIDTH , num_channels),padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))

        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))

        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
    
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        return model

    def customcnn_grex_plate_classifier(self, input_shape):
        HEIGHT, WIDTH, num_channels = input_shape
        model = Sequential()
        model.add(Conv2D(16, (3, 3), input_shape=(HEIGHT, WIDTH , num_channels), padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(128, (3, 3), padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        return model  

    def People_model_lite_skip(self, input_shape):
        HEIGHT, WIDTH, num_channels = input_shape
        image = Input(shape=input_shape)

        x = Conv2D(8, (3, 3), padding='valid')(image)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(8, (3, 3), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        skip1=x
        skip1 = Conv2D(8, (5, 5), padding='same')(skip1)
        skip1 = BatchNormalization()(skip1)
        skip1 = Activation('relu')(skip1)
        skip1 = MaxPooling2D(pool_size=(2, 2))(skip1)

        skip1 = Conv2D(8, (7, 7), padding='same')(skip1)
        skip1 = BatchNormalization()(skip1)
        skip1 = Activation('relu')(skip1)
        skip1 = MaxPooling2D(pool_size=(2, 2))(skip1)

        x = Conv2D(8, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(8, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = concatenate([x, skip1], axis=3)
        skip1=x



        
        return Model(inputs=image, outputs=x)

    def smaller_model(self,input_shape):
        #import pdb;pdb.set_trace()
        HEIGHT, WIDTH, num_channels = input_shape
        #input_shape = (128, 128, 3) # provide input shape
        #classes = 10 # replace classes with the number of classes
        input_flow = Input(shape=input_shape)
        x = SeparableConv2D(8, (3, 3), padding='same', activation='relu')(input_flow)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = SeparableConv2D(16, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(x)
        
        one = SeparableConv2D(8, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        one = SpatialDropout2D(0.25)(one)
        one = SeparableConv2D(8, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(one)
        two = SeparableConv2D(8, (5, 5), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        two = SpatialDropout2D(0.25)(two)
        two = SeparableConv2D(8, (5, 5), padding='same', activation='relu', dilation_rate=(2, 2))(two)
        
        x = concatenate([one, two])
        
        x = ActivityRegularization(l1=0.001)(x)
        x = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        one = SeparableConv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        one = GaussianDropout(0.25)(one)
        one = SeparableConv2D(32, (3, 3), padding='same', activation='relu', dilation_rate=(2, 2))(one)
        two = SeparableConv2D(32, (5, 5), padding='same', activation='relu', dilation_rate=(2, 2))(x)
        two = GaussianDropout(0.25)(two)
        two = SeparableConv2D(32, (5, 5), padding='same', activation='relu', dilation_rate=(2, 2))(two)

        concat = concatenate([one, two], axis=3)
        
        concat = ActivityRegularization(l2=0.001)(concat)
        concat = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(concat)
        x = Add()([x, concat])
        
        x = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        model = Model(inputs=input_flow, outputs=x)
        plot_model(model, show_shapes=True)
        model.summary()
        return model


    def depthwise_block(self,x,pointwise_conv_filters,oneone=True):
        x = ZeroPadding2D((1,1))(x)
        x = DepthwiseConv2D((3, 3), padding='valid', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if oneone is True:
            x = Conv2D(pointwise_conv_filters, (1, 1),
                              padding='valid',
                              use_bias=False,
                              strides=(1, 1))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        x = Dropout(.1)(x)
        return x
        
    def check_tensor_shape(self,x):
        k1 = x.shape[1:]
        k2=[int(x) for x in k1]
        buf = np.prod(k2)
        buf =(buf*4)/1024
        #print(x)
        print(x.name, k2,end='')
        print(', buf size:{:.2f}'.format(buf))
        #import pdb;pdb.set_trace()
        if buf > 230:
            print('buffer size is too large')
            #import pdb;pdb.set_trace()
            sys.exit()

        return x
    
    def People_model_lite(self,input_shape):
        HEIGHT, WIDTH, num_channels = input_shape
        input_flow = Input(shape=input_shape)
        y = Lambda(self.check_tensor_shape)(input_flow)
        x = Conv2D(4, (3, 3),strides=(2,2),dilation_rate=(1,1),padding='valid')(input_flow)
        y = Lambda(self.check_tensor_shape)(x)
        #x = MaxPooling2D((2, 2), padding='same')(x)
        x = Activation('relu')(x)

        x = Conv2D(8, (3, 3),strides=(1,1), padding='valid')(x)
        y = Lambda(self.check_tensor_shape)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Activation('relu')(x)
 
        
        x = self.depthwise_block(x,pointwise_conv_filters=8,oneone=False)
        x = self.depthwise_block(x,pointwise_conv_filters=8,oneone=False)
        y = Lambda(self.check_tensor_shape)(x)
        
        
        x = Conv2D(16, (3, 3),strides=(1,1),dilation_rate=(1,1), padding='valid')(x)
        y = Lambda(self.check_tensor_shape)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Activation('relu')(x)
        y = Lambda(self.check_tensor_shape)(x)
        
        x = self.depthwise_block(x,pointwise_conv_filters=16,oneone=False)
        x = self.depthwise_block(x,pointwise_conv_filters=16,oneone=False)
        y = Lambda(self.check_tensor_shape)(x)
        
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(32, (3, 3),strides=(1,1),dilation_rate=(1,1), padding='valid')(x)
        y = Lambda(self.check_tensor_shape)(x)
        x = Activation('relu')(x)
        
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(16, (3, 3),strides=(1,1),dilation_rate=(1,1), padding='valid')(x)
        y = Lambda(self.check_tensor_shape)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Activation('relu')(x)
        
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(16, (3, 3),strides=(1,1),dilation_rate=(1,1), padding='valid')(x)
        y = Lambda(self.check_tensor_shape)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Activation('relu')(x)
        y = Lambda(self.check_tensor_shape)(x)
        
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(8, (3, 3),dilation_rate=(1,1),padding='valid')(x)
        y = Lambda(self.check_tensor_shape)(x)
        x = Activation('relu')(x)
        
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(2, (3, 3),dilation_rate=(1,1),padding='valid')(x)
        y = Lambda(self.check_tensor_shape)(x)
        x = Activation('relu')(x)
        
        model = Model(inputs=input_flow, outputs=x)
        #plot_model(model, show_shapes=True)
        #model.summary()
        return model
        
    
    def People_model_lite13(self, input_shape):
        HEIGHT, WIDTH, num_channels = input_shape
        model = Sequential()

        model.add(Conv2D(2, (3, 3),strides=(2,2), input_shape=(HEIGHT, WIDTH , num_channels),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.1))
        
        x = ZeroPadding2D((1,1))(x)
        model.add(Conv2D(4, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.1))
        model.add(ZeroPadding2D(padding=(1, 1)))
    
        model.add(DepthwiseConv2D((3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.1))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        #model.add(depthwise_block(x,pointwise_conv_filters)
        
        model.add(SeparableConv2D(4, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.1))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        
        model.add(Conv2D(8, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.1))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        model.add(SeparableConv2D(8, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.1))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        model.add(SeparableConv2D(8, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.1))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        model.add(SeparableConv2D(8, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.1))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        model.add(Conv2D(4, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.1))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        
        '''
        
        model.add(DepthwiseConv2D(4, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.1))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        model.add(DepthwiseConv2D(4, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.1))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        model.add(DepthwiseConv2D(4, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.1))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        
        model.add(DepthwiseConv2D(8, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(.1))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        model.add(DepthwiseConv2D(8, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.1))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        model.add(DepthwiseConv2D(16, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.1))
        model.add(ZeroPadding2D(padding=(1, 1)))
        
        model.add(DepthwiseConv2D(32, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(.1))
        model.add(ZeroPadding2D(padding=(1, 1)))
        '''
        

        model.add(Conv2D(2, (3, 3),padding='valid'))
        #model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        return model




    def custom_people_model_lite(self, input_shape):
        HEIGHT, WIDTH, num_channels = input_shape
        model = Sequential()    
        model.add(Conv2D(2,(3, 3), input_shape=(HEIGHT, WIDTH , num_channels),padding='valid')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

        model.add(Conv2D(4,(3, 3),padding='same')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
        
        model.add(Conv2D(8,(3, 3),padding='same'))
        model.add(Activation('relu'))
    
        model.add(Conv2D(8,(3, 3),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    
        model.add(Conv2D(16,(3, 3),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))

        model.add(Conv2D(32,(3, 3),padding='same'))
        model.add(Activation('relu'))

        return model


    def People_model_lite5(self, input_shape):
        HEIGHT, WIDTH, num_channels = input_shape
        model = Sequential()
        import random

        model.add(Conv2D(1, (1, 1), input_shape=(HEIGHT, WIDTH , num_channels),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        while(1):
            try:
                k1 = random.randint(4, 10)
                for i in range(k1):
                    k = random.randint(0, 12)
                    if k>3:
                        model.add(Conv2D(k, (3, 3),padding='valid'))
                        model.add(BatchNormalization())
                        model.add(Activation('relu'))
                        k = random.randint(0, 12)
                        if k>3:
                            model.add(MaxPooling2D(pool_size=(2, 2)))
                break
            except:
                pass

        return model


    def RX_model_digit(self, input_shape):  
        HEIGHT, WIDTH, num_channels = input_shape
        model = Sequential()
        model.add(Conv2D(8, (3, 3), input_shape=(HEIGHT, WIDTH , num_channels),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(8, (3, 3),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (3, 3),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(10, (3, 3),padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        return model

    def RX_model_gear(self, input_shape):
        HEIGHT, WIDTH, num_channels = input_shape
        model = Sequential()
        model.add(Conv2D(4, (3, 3), input_shape=(HEIGHT, WIDTH, num_channels),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(4, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(4, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(4, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(Conv2D(2, (3, 3),padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        return model

    
    def regressioncnn(self, input_shape):
        HEIGHT, WIDTH, num_channels = input_shape
        image = Input(shape=input_shape)
        
        locnet = Conv2D(4, (3, 3), padding='same')(image)
        locnet = BatchNormalization()(locnet)
        locnet = MaxPool2D(pool_size=(2, 2))(locnet)
        locnet = Activation('relu')(locnet)
        
        locnet = Conv2D(20, (3, 3), padding='same')(locnet)
        locnet = BatchNormalization()(locnet)
        locnet = MaxPool2D(pool_size=(2, 2))(locnet)
        locnet = Activation('relu')(locnet)
        
        locnet = Conv2D(20, (3, 3), padding='same')(locnet)
        locnet = BatchNormalization()(locnet)
        locnet = MaxPool2D(pool_size=(2, 2))(locnet)
        locnet = Activation('relu')(locnet)
        
        locnet = Conv2D(40, (3, 3), padding='same')(locnet)
        locnet = BatchNormalization()(locnet)
        locnet = Activation('relu')(locnet)
        
        locnet = Conv2D(40, (3, 3), padding='same')(locnet)
        locnet = BatchNormalization()(locnet)
        locnet = Activation('relu')(locnet)
        return Model(inputs=image, outputs=locnet)

    def regressioncnn_skip(self, input_shape):
        HEIGHT, WIDTH, num_channels = input_shape
        image = Input(shape=input_shape)
        
        locnet = Conv2D(20, (3, 3), padding='same')(image)
        locnet = BatchNormalization()(locnet)
        locnet = MaxPool2D(pool_size=(2, 2))(locnet)
        locnet = Activation('relu')(locnet)
        
        locnet = Conv2D(20, (3, 3), padding='same')(locnet)
        locnet = BatchNormalization()(locnet)
        locnet = MaxPool2D(pool_size=(2, 2))(locnet)
        locnet = Activation('relu')(locnet)
        
        locnet = Conv2D(20, (3, 3), padding='same')(locnet)
        locnet = BatchNormalization()(locnet)
        locnet = MaxPool2D(pool_size=(2, 2))(locnet)
        locnet = Activation('relu')(locnet)
        
        locnet = Conv2D(40, (3, 3), padding='same')(locnet)
        locnet = BatchNormalization()(locnet)
        locnet = Activation('relu')(locnet)
        
        locnet = Conv2D(40, (3, 3), padding='same')(locnet)
        locnet = BatchNormalization()(locnet)
        locnet = Activation('relu')(locnet)
        return Model(inputs=image, outputs=locnet)
    
        
if __name__=='__main__':
    path = osp.dirname(osp.abspath(__file__))
    sys.path.append(path)
    sys.path.append(os.path.join(path, '..', 'ign_utils'))
    
    model_obj   = MODEL(inp_dim=(8000, 1),
        modelname="Conv1DModel",
        freeze=False, 
        num_unfreeze=5, 
        pooling_layer=None,
        dropout=[0.5], 
        fc_layers=[256], 
        num_outnodes=12, 
        mode='categorical',
        load_prev_weights=False
    )

    model, preprocessing_function = model_obj.get_model()



