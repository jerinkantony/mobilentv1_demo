# mobilentv1_demo

```git clone https://github.com/jerinkantony/mobilentv1_demo --depth=1```

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

## ```Step3:To Generate quantized models and testing```
    
    Run `python tflite_digitmodule.py`  - for generating NPR_digit_class quantized weights  

# C. To view Results

## ```Check train_log.txt for CNN results```

## ```Check quant_log.txt for Quantized results```


# D. To run sample testing with single image

## ```Run sample_quant_prediction.py```





