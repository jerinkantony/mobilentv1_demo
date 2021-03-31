#load model
import numpy as np
import tensorflow as tf
import glob
import os
import os.path as osp
from keras.models import load_model
from keras.preprocessing import image
from validation_scores import topn_accuracy, mean_avg_precision
from sklearn import metrics


def load_predict_fp32(img_dir,model_dir):
    model = load_model(model_dir)
    allfiles = glob.glob(osp.join(img_dir, '*', '*'), recursive=True)
    pred_list = []
    probs_list = []
    actual_list = []
    for f in allfiles:
        img = image.load_img(f, target_size=(64, 64), grayscale=True)
        img_tensor = image.img_to_array(img)
        img_tensor = img_tensor.astype('float32')                 
        img_tensor = np.expand_dims(img_tensor, axis=0)         
        img_tensor /= 255.
        probs = model.predict(img_tensor).tolist()[0]
        probs_list.append(probs)
        predict_label = np.argmax(probs)
        pred_list.append(int(predict_label))
        classname = os.path.split(f)[0].split('/')[-1]
        actual_list.append(int(classname))
        print("Predicted ", predict_label, "Actual ", classname)
        
    return probs_list,pred_list, actual_list

img_path = osp.join('../DB/watermeter_digit_class/val')
model_path = '../weights/watermeter_digit_class/RX_model.h5'
probs_list,pred_list, actual_list = load_predict_fp32(img_path, model_path)
top1_acc = topn_accuracy(probs_list, actual_list,1)
top5_acc = topn_accuracy(probs_list, actual_list,5)
mean_av_prec = mean_avg_precision(pred_list, actual_list)
f1_acc = print("F1 score:",metrics.f1_score(actual_list, pred_list, average='macro'))


