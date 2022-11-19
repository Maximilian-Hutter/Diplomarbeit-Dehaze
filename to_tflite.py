import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
 
onnx_model = onnx.load("./Dehaze.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("./Dehaze.pb")

# make a converter object from the saved tensorflow file
converter = tf.lite.TFLiteConverter.from_saved_model('./Dehaze.pb')
# tell converter which type of optimization techniques to use
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

# convert the model 
tf_lite_model = converter.convert()
# save the converted model 
open('./Dehaze.tflite', 'wb').write(tf_lite_model)