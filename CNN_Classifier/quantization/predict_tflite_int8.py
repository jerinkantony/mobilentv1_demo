import numpy as np
import tensorflow as tf
import glob
import os
import os.path as osp
from keras.preprocessing import image
from sklearn import metrics
from validation_scores import topn_accuracy, mean_avg_precision
def load_predict_int8(img_dir,model_dir):
    
    allfiles = glob.glob(osp.join(img_dir, '*', '*'), recursive=True) #other exts TOTO
    interpreter_quant = tf.lite.Interpreter(model_dir)
    interpreter_quant.allocate_tensors()
    input_index = interpreter_quant.get_input_details()[0]["index"]
    output_index = interpreter_quant.get_output_details()[0]["index"]
    good_count=0
    pred_list = []
    actual_list = []
    probs_list = []
    for f in allfiles:
	#load image
        img = image.load_img(f, target_size=(64, 64), grayscale=True)
        img_tensor = image.img_to_array(img)
        img_tensor = img_tensor.astype('float32')                 
        img_tensor = np.expand_dims(img_tensor, axis=0)         
        img_tensor /= 255. 
        
	#load image to interpreter
        interpreter_quant.set_tensor(input_index, img_tensor)
        interpreter_quant.invoke()
        predictions = interpreter_quant.get_tensor(output_index).tolist()[0]
        
	#get the predictions
        classname = os.path.split(f)[0].split('/')[-1]
        predicted_labels = np.argmax(predictions, axis=-1)
        pred_list.append(int(predicted_labels))
        probs_list.append(predictions)
        actual_list.append(int(classname))
        print('Predicted: ', predicted_labels, 'Actual: ', classname)
    
    return probs_list,pred_list, actual_list




img_path = osp.join('../DB/watermeter_digit_class/val')
model_path = '../weights/watermeter_digit_class/RX_model_quant.tflite'
probs_list,pred_list, actual_list = load_predict_int8(img_path, model_path)
top1_acc = topn_accuracy(probs_list, actual_list,1)
top5_acc = topn_accuracy(probs_list, actual_list,5)
mean_av_prec = mean_avg_precision(pred_list, actual_list)
f1_acc = print("F1 score:",metrics.f1_score(actual_list, pred_list, average='macro'))

