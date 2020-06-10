# From https://github.com/openai/improved-gan/blob/master/inception_score/model.py
# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
# tf.config.gpu.set_per_process_memory_growth(True)
import glob
import scipy.misc
import math
import time

class IS(object):
  def __init__(self, MODEL_DIR):
    self.MODEL_DIR = MODEL_DIR # '/DataSet/imagenet_inceptionNet'
    self.DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    self._init_inception()

  # Call this function with list of images. Each of elements should be a 
  # numpy array with values ranging from 0 to 255.
  def get_inception_score(self,images, splits=10):
    # assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
      img = img.astype(np.float32)
      inps.append(np.expand_dims(img, 0))
    bs = 50
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      preds = []
      n_batches = int(math.ceil(float(len(inps)) / float(bs)))
      for i in range(n_batches):
          # sys.stdout.write(".")
          # sys.stdout.flush()
          inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
          inp = np.concatenate(inp, 0)
          pred = sess.run(self.softmax, {'InputTensor:0': inp})
          preds.append(pred)
      preds = np.concatenate(preds, 0)
      scores = []
      for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
      return np.mean(scores), np.std(scores)

  # This function is called automatically.
  def _init_inception(self):
    # global softmax
    if not os.path.exists(self.MODEL_DIR):
      os.makedirs(self.MODEL_DIR)
    filename = self.DATA_URL.split('/')[-1]
    filepath = os.path.join(self.MODEL_DIR, filename)
    if not os.path.exists(filepath):
      os.makedirs(os.path.join(self.MODEL_DIR, filename+"_occupied"))
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(self.DATA_URL, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
      os.rmdir(os.path.join(self.MODEL_DIR, filename+"_occupied"))
    while(os.path.exists(os.path.join(self.MODEL_DIR, filename+"_occupied"))):
      time.sleep(60)
    tarfile.open(filepath, 'r:gz').extractall(self.MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
        self.MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      # _ = tf.import_graph_def(graph_def, name='')
      # Import model with a modification in the input tensor to accept arbitrary
      # batch size.
      input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                    name='InputTensor')
      _ = tf.import_graph_def(graph_def, name='',
                              input_map={'ExpandDims:0':input_tensor})
    # Works with an arbitrary minibatch size.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      pool3 = sess.graph.get_tensor_by_name('pool_3:0')
      ops = pool3.graph.get_operations()
      for op_idx, op in enumerate(ops):
          for o in op.outputs:
              shape = o.get_shape()
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                  if s == 1 and j == 0:
                      new_shape.append(None)
                  else:
                      new_shape.append(s)
              o.set_shape(tf.TensorShape(new_shape))
      w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
      logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
      self.softmax = tf.nn.softmax(logits)

  # if softmax is None:
  #   _init_inception()