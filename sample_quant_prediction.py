import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import glob
import os
import os.path as osp
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
import warnings 
warnings.filterwarnings('ignore')

test_path = "CNN_Classifier/DB/japanese_NPR_digit_class/val"
preprocessing_function = preprocess_input
img_width,img_height,ch = 128,128,3

def representative_data_gen():

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(test_path,target_size = (img_width,img_height),batch_size=1,shuffle=False,class_mode='categorical')
    for ind in range(len(test_generator.filenames)):
        img_with_label = test_generator.next()
    yield [np.array(img_with_label[0],dtype=np.float32)]

def pre_process(img):
    if ch==1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_width,img_height)) #, interpolation=cv2.INTER_NEAREST
    if not preprocessing_function is None and ch==3:
        img =  preprocessing_function(img)
        img = img.astype(np.float32)
    else:
        img = img.astype(np.float32)
        img = img / 255.       
    if ch == 1:
        img =np.expand_dims(img,axis=2)
    return img

def test_model(val_img,tf_path,tflite_path): 

    dataset_labels = ['2', '7', '00', '1', '90', '0', '5', '4', '3', '8', '9', '6', '13', '18', '22', '23', '33', '100']      
    val_img = pre_process(val_image)
    Original_label = os.path.split(os.path.split(img)[0])[1]
    val_image1 = np.expand_dims(val_img, axis=0)
    val_image2 = val_image1*255
    val_image2 = val_image2.astype(np.uint8)

    keras_model = load_model(tf_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model_quant = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], val_image2)
    interpreter.invoke()
    tflitequantised_model_predictions = interpreter.get_tensor(output_details[0]['index']) 
    tflitequantised_predicted_ids = np.argmax(tflitequantised_model_predictions, axis=-1)
    tflitequantised_predicted_ids = dataset_labels[int(tflitequantised_predicted_ids)]
    print('TFLite Quantised : Actual: ', Original_label, 'Predicted: ', tflitequantised_predicted_ids)

if __name__ == '__main__':

    tf_path = "CNN_Classifier/weights/japanese_NPR_digit_class/MobileNet.h5"
        
    tflite_path= "CNN_Classifier/weights/japanese_NPR_digit_class/converted_model_float32.tflite"

    tflite_quantized_path= "CNN_Classifier/weights/japanese_NPR_digit_class/converted_model_int8quantized.tflite"

    if 1: 
        img = "CNN_Classifier/DB/japanese_NPR_digit_class/val/0/img001-00006.png"
        val_image = cv2.imread(img)
        test_model(val_image,tf_path,tflite_path)
        cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
        cv2.imshow('Image', val_image)
        cv2.waitKey(0)



