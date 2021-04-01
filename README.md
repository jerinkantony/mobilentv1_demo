# ```mobilentv1_demo```

 git clone https://github.com/jerinkantony/mobilentv1_demo --depth=1

# A. Training

## ```Step1:Create Virtual enviornment and activate```

    cd CNN_Classifier

    virtualenv venv_train --python=python3

    source venv_train/bin/activate


## ```Step2:Install Requirements```

    pip install -r requirements.txt

## ```Step3:Train and Validate```

    python cnn_demo.py


# B. Quantization

## ```Step1:Create Virtual enviornment and activate```

    cd ../

    deactivate(deactivate training enviornment)

    virtualenv venv_quant --python=python3

    source venv_quant/bin/activate


## ```Step2:Install Requirements```

    pip install -r requirements.txt

##  ```Step3:To Generate quantized models and testing```
    
    python tflite_digitmodule.py  - for generating NPR_digit_class quantized weights  

# C. ```To view Results```

Check train_log.txt for training logs

Check quant_log.txt for Tflite and Quantization logs


# D. ```To run sample testing with single image```

 python sample_quant_prediction.py


# E. ```Model Summary```

_________________________________________________________________
Layer (type)                 Output Shape              Param #

input_2 (InputLayer)         (None, 128, 128, 3)       0         
_________________________________________________________________
conv1_pad (ZeroPadding2D)    (None, 129, 129, 3)       0         
_________________________________________________________________
conv1 (Conv2D)               (None, 64, 64, 32)        864       
_________________________________________________________________
conv1_bn (BatchNormalization (None, 64, 64, 32)        128       
_________________________________________________________________
conv1_relu (ReLU)            (None, 64, 64, 32)        0         
_________________________________________________________________
conv_dw_1 (DepthwiseConv2D)  (None, 64, 64, 32)        288       
_________________________________________________________________
conv_dw_1_bn (BatchNormaliza (None, 64, 64, 32)        128       
_________________________________________________________________
conv_dw_1_relu (ReLU)        (None, 64, 64, 32)        0         
_________________________________________________________________
conv_pw_1 (Conv2D)           (None, 64, 64, 64)        2048      
_________________________________________________________________
conv_pw_1_bn (BatchNormaliza (None, 64, 64, 64)        256       
_________________________________________________________________
conv_pw_1_relu (ReLU)        (None, 64, 64, 64)        0         
_________________________________________________________________
conv_pad_2 (ZeroPadding2D)   (None, 65, 65, 64)        0         
_________________________________________________________________
conv_dw_2 (DepthwiseConv2D)  (None, 32, 32, 64)        576       
_________________________________________________________________
conv_dw_2_bn (BatchNormaliza (None, 32, 32, 64)        256       
_________________________________________________________________
conv_dw_2_relu (ReLU)        (None, 32, 32, 64)        0         
_________________________________________________________________
conv_pw_2 (Conv2D)           (None, 32, 32, 128)       8192      
_________________________________________________________________
conv_pw_2_bn (BatchNormaliza (None, 32, 32, 128)       512       
_________________________________________________________________
conv_pw_2_relu (ReLU)        (None, 32, 32, 128)       0         
_________________________________________________________________
conv_dw_3 (DepthwiseConv2D)  (None, 32, 32, 128)       1152      
_________________________________________________________________
conv_dw_3_bn (BatchNormaliza (None, 32, 32, 128)       512       
_________________________________________________________________
conv_dw_3_relu (ReLU)        (None, 32, 32, 128)       0         
_________________________________________________________________
conv_pw_3 (Conv2D)           (None, 32, 32, 128)       16384     
_________________________________________________________________
conv_pw_3_bn (BatchNormaliza (None, 32, 32, 128)       512       
_________________________________________________________________
conv_pw_3_relu (ReLU)        (None, 32, 32, 128)       0         
_________________________________________________________________
conv_pad_4 (ZeroPadding2D)   (None, 33, 33, 128)       0         
_________________________________________________________________
conv_dw_4 (DepthwiseConv2D)  (None, 16, 16, 128)       1152      
_________________________________________________________________
conv_dw_4_bn (BatchNormaliza (None, 16, 16, 128)       512       
_________________________________________________________________
conv_dw_4_relu (ReLU)        (None, 16, 16, 128)       0         
_________________________________________________________________
conv_pw_4 (Conv2D)           (None, 16, 16, 256)       32768     
_________________________________________________________________
conv_pw_4_bn (BatchNormaliza (None, 16, 16, 256)       1024      
_________________________________________________________________
conv_pw_4_relu (ReLU)        (None, 16, 16, 256)       0         
_________________________________________________________________
conv_dw_5 (DepthwiseConv2D)  (None, 16, 16, 256)       2304      
_________________________________________________________________
conv_dw_5_bn (BatchNormaliza (None, 16, 16, 256)       1024      
_________________________________________________________________
conv_dw_5_relu (ReLU)        (None, 16, 16, 256)       0         
_________________________________________________________________
conv_pw_5 (Conv2D)           (None, 16, 16, 256)       65536     
_________________________________________________________________
conv_pw_5_bn (BatchNormaliza (None, 16, 16, 256)       1024      
_________________________________________________________________
conv_pw_5_relu (ReLU)        (None, 16, 16, 256)       0         
_________________________________________________________________
conv_pad_6 (ZeroPadding2D)   (None, 17, 17, 256)       0         
_________________________________________________________________
conv_dw_6 (DepthwiseConv2D)  (None, 8, 8, 256)         2304      
_________________________________________________________________
conv_dw_6_bn (BatchNormaliza (None, 8, 8, 256)         1024      
_________________________________________________________________
conv_dw_6_relu (ReLU)        (None, 8, 8, 256)         0         
_________________________________________________________________
conv_pw_6 (Conv2D)           (None, 8, 8, 512)         131072    
_________________________________________________________________
conv_pw_6_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_pw_6_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_dw_7 (DepthwiseConv2D)  (None, 8, 8, 512)         4608      
_________________________________________________________________
conv_dw_7_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_dw_7_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_pw_7 (Conv2D)           (None, 8, 8, 512)         262144    
_________________________________________________________________
conv_pw_7_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_pw_7_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_dw_8 (DepthwiseConv2D)  (None, 8, 8, 512)         4608      
_________________________________________________________________
conv_dw_8_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_dw_8_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_pw_8 (Conv2D)           (None, 8, 8, 512)         262144    
_________________________________________________________________
conv_pw_8_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_pw_8_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_dw_9 (DepthwiseConv2D)  (None, 8, 8, 512)         4608      
_________________________________________________________________
conv_dw_9_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_dw_9_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_pw_9 (Conv2D)           (None, 8, 8, 512)         262144    
_________________________________________________________________
conv_pw_9_bn (BatchNormaliza (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_pw_9_relu (ReLU)        (None, 8, 8, 512)         0         
_________________________________________________________________
conv_dw_10 (DepthwiseConv2D) (None, 8, 8, 512)         4608      
_________________________________________________________________
conv_dw_10_bn (BatchNormaliz (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_dw_10_relu (ReLU)       (None, 8, 8, 512)         0         
_________________________________________________________________
conv_pw_10 (Conv2D)          (None, 8, 8, 512)         262144    
_________________________________________________________________
conv_pw_10_bn (BatchNormaliz (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_pw_10_relu (ReLU)       (None, 8, 8, 512)         0         
_________________________________________________________________
conv_dw_11 (DepthwiseConv2D) (None, 8, 8, 512)         4608      
_________________________________________________________________
conv_dw_11_bn (BatchNormaliz (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_dw_11_relu (ReLU)       (None, 8, 8, 512)         0         
_________________________________________________________________
conv_pw_11 (Conv2D)          (None, 8, 8, 512)         262144    
_________________________________________________________________
conv_pw_11_bn (BatchNormaliz (None, 8, 8, 512)         2048      
_________________________________________________________________
conv_pw_11_relu (ReLU)       (None, 8, 8, 512)         0         
_________________________________________________________________
conv_pad_12 (ZeroPadding2D)  (None, 9, 9, 512)         0         
_________________________________________________________________
conv_dw_12 (DepthwiseConv2D) (None, 4, 4, 512)         4608      
_________________________________________________________________
conv_dw_12_bn (BatchNormaliz (None, 4, 4, 512)         2048      
_________________________________________________________________
conv_dw_12_relu (ReLU)       (None, 4, 4, 512)         0         
_________________________________________________________________
conv_pw_12 (Conv2D)          (None, 4, 4, 1024)        524288    
_________________________________________________________________
conv_pw_12_bn (BatchNormaliz (None, 4, 4, 1024)        4096      
_________________________________________________________________
conv_pw_12_relu (ReLU)       (None, 4, 4, 1024)        0         
_________________________________________________________________
conv_dw_13 (DepthwiseConv2D) (None, 4, 4, 1024)        9216      
_________________________________________________________________
conv_dw_13_bn (BatchNormaliz (None, 4, 4, 1024)        4096      
_________________________________________________________________
conv_dw_13_relu (ReLU)       (None, 4, 4, 1024)        0         
_________________________________________________________________
conv_pw_13 (Conv2D)          (None, 4, 4, 1024)        1048576   
_________________________________________________________________
conv_pw_13_bn (BatchNormaliz (None, 4, 4, 1024)        4096      
_________________________________________________________________
conv_pw_13_relu (ReLU)       (None, 4, 4, 1024)        0         
_________________________________________________________________
global_average_pooling2d_2 ( (None, 1024)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 16)                16400     
_________________________________________________________________
dropout_2 (Dropout)          (None, 16)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 28)                476       
=================================================================
Total params: 3,245,740
Trainable params: 3,223,852
Non-trainable params: 21,888
_________________________________________________________________


