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
img_width,img_height,ch = 128,128,3
test_path = "CNN_Classifier/DB/japanese_NPR_color_class/val"

def representative_data_gen():

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(test_path,target_size = (img_width,img_height),batch_size=1,shuffle=False,class_mode='categorical')
    for ind in range(len(test_generator.filenames)):
        img_with_label = test_generator.next()
    yield [np.array(img_with_label[0],dtype=np.float32)]

def convert_tf_to_tflie(tf_path,tflite_path):
    keras_model = load_model(tf_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)

def check_quantised_model(tflite_model_quant):
    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    print(interpreter)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)

def convert_tf_to_tflite_int8(tf_path,tflite_path,tflite_quantized_path):

    keras_model = load_model(tf_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_without_quant = converter.convert()
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model_quant = converter.convert()
    open(tflite_quantized_path,'wb').write(tflite_model_quant)
    open(tflite_path,'wb').write(tflite_without_quant)
    return tflite_model_quant,tflite_without_quant

def acc_matrics(y_true, y_pred,labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print('Confusion matrix:\n',cm)
    print('Report :\n',classification_report(y_true, y_pred))
    goodcount=0
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            goodcount+=1    
    print('Val Acc: ', goodcount/len(y_true) * 100, '%\n\n')

def pre_process(img):
    img = cv2.resize(img, (img_width, img_height))
    if ch==1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #if not preprocessing_function is None and self.ch==3:
    #img = preprocess_input(img)
    #, interpolation=cv2.INTER_NEAREST
    img = img.astype(np.float32)
    img = img / 255.
    return img

def test_models(test_path,tf_path,tflite_model_quant,tflite_path):

    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    interpreter.allocate_tensors()
    keras_model = load_model(tf_path)
    tflite_y_true=[]
    tflite_y_pred=[]
    tf_y_true =[]
    tf_y_pred =[]
    tflitequantised_y_pred =[]
    tflitequantised_y_true = []
    
   
    dataset_labels =  [
              "black",
              "white",
              "green",
              "yellow"
            ]

    allfiles = glob.glob(osp.join(test_path, '*', '*'), recursive=True)
    for img in allfiles: 
        val_image = cv2.imread(img)
        print(img)
        val_img = pre_process(val_image)
        Original_label = os.path.split(os.path.split(img)[0])[1]
        val_image1 = np.expand_dims(val_img, axis=0)
        val_image2 = val_image1*255
        val_image2 = val_image2.astype(np.uint8)
        #val_image2 = np.expand_dims(val_image2, axis=0)
       
        #TF Lite model
      
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], val_image2)
        interpreter.invoke()
        tflitequantised_model_predictions = interpreter.get_tensor(output_details[0]['index']) 
        tflitequantised_predicted_ids = np.argmax(tflitequantised_model_predictions, axis=-1)
        tflitequantised_predicted_ids = dataset_labels[int(tflitequantised_predicted_ids)]
        print('TFLite Quantised : Actual: ', Original_label, 'Predicted: ', tflitequantised_predicted_ids)
        tflitequantised_y_pred.append(tflitequantised_predicted_ids)
        tflitequantised_y_true.append(Original_label)

        #tflite float32

        interpreter1 = tf.lite.Interpreter(model_path=tflite_path)
        interpreter1.allocate_tensors()
        input_details1 = interpreter1.get_input_details()
        output_details1 = interpreter1.get_output_details()
        interpreter1.set_tensor(input_details1[0]['index'], val_image1)
        interpreter1.invoke()
        tflite_model_predictions = interpreter1.get_tensor(output_details1[0]['index']) 
        tflite_predicted_ids = np.argmax(tflite_model_predictions, axis=-1)
        tflite_predicted_ids = dataset_labels[int(tflite_predicted_ids)]
        print('TFLite : Actual: ', Original_label, 'Predicted: ', tflite_predicted_ids)
        tflite_y_pred.append(tflite_predicted_ids)
        tflite_y_true.append(Original_label)
        
        #tensorflow model

        tf_prediction = keras_model.predict(val_image1).tolist()[0]
        tf_predicted_ids = np.argmax(tf_prediction)
        tf_predicted_labels = dataset_labels[tf_predicted_ids]
        print('TF : Actual: ', Original_label, 'Predicted: ', tf_predicted_labels)
        tf_y_pred.append(tf_predicted_labels)
        tf_y_true.append(Original_label)

    print("TFLite Model")
    acc_matrics(tflite_y_true,tflite_y_pred,dataset_labels)
    print("TF Model")
    acc_matrics(tf_y_true,tf_y_pred,dataset_labels)

if __name__ == '__main__':

    
    tf_path = "CNN_Classifier/weights/japanese_NPR_color_class/MobileNet.h5"
        
    tflite_path= "CNN_Classifier/weights/japanese_NPR_color_class/converted_model_float32.tflite"

    tflite_quantized_path= "CNN_Classifier/weights/japanese_NPR_color_class/converted_model_int8quantized.tflite"

    if 0:#Convert tf to tflite without quantized

        convert_tf_to_tflie(tf_path,tflite_path)
    
 
    if 1: #Convert tf to tflite with quantized
        tflite_model_quant,tflite_without_quant = convert_tf_to_tflite_int8(tf_path,tflite_path,tflite_quantized_path)
        check_quantised_model(tflite_model_quant)#check dtype of quantised model
        if 1:#Testing

            test_path = "CNN_Classifier/DB/japanese_NPR_color_class/val"
            test_models(test_path,tf_path,tflite_model_quant,tflite_path)



