import tensorflow as tf
import pathlib
import os.path
def quantize_model(model_path, out_dir):
	converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
	print("converting model into tflite...")
	tflite_model = converter.convert()
	tflite_models_dir = pathlib.Path(out_dir)
	tflite_models_dir.mkdir(exist_ok=True, parents=True)
	tflite_model_file = tflite_models_dir/"RX_model.tflite"
	tflite_model_file.write_bytes(tflite_model)
	print("Quantizing model...")
	converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
	tflite_quant_model = converter.convert()
	tflite_model_quant_file = tflite_models_dir/"RX_model_quant.tflite"
	tflite_model_quant_file.write_bytes(tflite_quant_model)
	return tflite_model_quant_file,tflite_models_dir
	

model_path = '../weights/watermeter_digit_class/RX_model.h5'
out_dir = '../weights/watermeter_digit_class/' 
quantized_model, tflite_models_dir = quantize_model(model_path, out_dir)
if os.path.exists(tflite_models_dir/"RX_model_quant.tflite") == True:
    print("Model quantized and saved to: ", tflite_models_dir)




		
