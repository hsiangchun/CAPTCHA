# CNN-LSTM-CTC-OCR
# Copyright (C) 2017, 2018 Jerod Weinman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import time

import numpy as np
import difflib

from PIL import Image

import tensorflow as tf
from tensorflow.contrib import learn

import mjsynth
import model
from lexicon import dictionary_from_file

import cv2
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

WINDOW_SHAPE = (128, 64)



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('val','../data/val_final',
                          """Directory for val data""")
tf.app.flags.DEFINE_string('model','../data/model',
                          """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('save','../data/val_final/partition/',
                          """Directory for Save make_neg pics""")
tf.app.flags.DEFINE_string('device','/gpu:0',
                           """Device for graph placement""")
tf.app.flags.DEFINE_string('lexicon','',
			   """File containing lexicon of image words""")


tf.logging.set_verbosity(tf.logging.WARN)

# Non-configurable parameters
mode = learn.ModeKeys.INFER # 'Configure' training mode for dropout layers

def _resize_image(image):
	"""[31*?]"""
	(x, y) = image.size
	x_s = int(31.0 * x / y)
	y_s = 31
	resizeimage = image.resize((x_s, y_s))
		
	return resizeimage

def make_scaled_ims(im, min_shape):
    yield im
    ratio = 1. / 2 ** 0.5
    (x, y) = im.size
    shape = (x / ratio, y / ratio)
    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))
        
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        yield im.resize(shape)
        
def _resize_image(image):
    """[31*?]"""
    (x, y) = image.size
    x_s = int(31.0 * x / y)
    y_s = 31
    resizeimage = image.resize((x_s, y_s))
		
    return resizeimage

def make_neg(filename,k,image,croph,cropw,movew,moveh):
    resize_img = _resize_image(image)
    #resize_img.save("test/" + str(k) + ".jpg")
    resize_img = np.array(resize_img)
    resize_img = resize_img[:,:,:1]
    all_image.append([k,resize_img])    
    k = k + 1
    (cols, rows) = image.size
    r1, r2 =  [0, croph] 
    while r2 <= rows:
        c1, c2 = [0, cropw] 
        while c2 <= cols:
            #  left, upper, right, and lower
            region = (c1 , r1, c2 , r2) 
            cropImg = image.crop(region)
            name = str(filename).split(".jpg")[0].split("-")[0]
            cropImg.save(FLAGS.save + str(name) + "-" + str(k)+ ".jpg")
            resize_img = _resize_image(cropImg)
            resize_img = np.array(resize_img)
            resize_img = resize_img[:,:,:1]
            all_image.append([k,resize_img])
            k, c1, c2 = [k + 1, c1 + movew, c2 + movew]
        r1, r2 = [r1 + moveh, r2 + moveh] 
    return k


def _get_image(k,filename):
    """Load image data for placement in graph"""
    image = Image.open(os.path.join(FLAGS.val,filename))
    ims = list(make_scaled_ims(image,WINDOW_SHAPE))
	
    for i in range(len(ims)):
        k = make_neg(filename, k, ims[i], croph = 40, cropw = 100, movew = 25,moveh = 8)
    return k

def _preprocess_image(image):

    # Copied from mjsynth.py. Should be abstracted to a more general module.
    
    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(image, 0.5)

    # Pad with copy of first row to expand to 32 pixels height
    first_row = tf.slice(image, [0, 0, 0], [1, -1, -1])
    image = tf.concat([first_row, image], 0)

    return image


def _get_input():
    """Set up and return image and width placeholder tensors"""

    # Raw image as placeholder to be fed one-by-one by dictionary
    image = tf.placeholder(tf.uint8, shape=[31, None, 1])
    width = tf.placeholder(tf.int32,shape=[]) # for ctc_loss

    return image,width


def _get_output(rnn_logits,sequence_length):
    """Create ops for validation
       predictions: Results of CTC beacm search decoding
    """
    with tf.name_scope("test"):
	if FLAGS.lexicon:
	    dict_tensor = _get_dictionary_tensor(FLAGS.lexicon, mjsynth.out_charset)
	    predictions,_ = tf.nn.ctc_beam_search_decoder_trie(rnn_logits,
	    					   sequence_length,
	    					   alphabet_size=mjsynth.num_classes() ,
	    					   dictionary=dict_tensor,
	    					   beam_width=128,
	    					   top_paths=1,
	    					   merge_repeated=True)
	else:
	    predictions,_ = tf.nn.ctc_beam_search_decoder(rnn_logits,
	    					   sequence_length,
	    					   beam_width=128,
	    					   top_paths=1,
	    					   merge_repeated=True)
    return predictions


def _get_session_config():
    """Setup session config to soften device placement"""
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config


def _get_checkpoint():
    """Get the checkpoint path from the given model output directory"""
    ckpt = tf.train.get_checkpoint_state(FLAGS.model)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path=ckpt.model_checkpoint_path
    else:
        raise RuntimeError('No checkpoint file found')

    return ckpt_path


def _get_init_trained():
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_STEP) + 
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )
    
    init_fn = lambda sess,ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn

def _get_string(labels):
    """Transform an 1D array of labels into the corresponding character string"""
    string = ''.join([mjsynth.out_charset[c] for c in labels])
    return string

def _get_dictionary_tensor(dictionary_path, charset):
    return tf.sparse_tensor_to_dense(tf.to_int32(
	dictionary_from_file(dictionary_path, charset)))

def main(argv=None):

    with tf.Graph().as_default():
        image,width = _get_input() # Placeholder tensors

        proc_image = _preprocess_image(image)
        proc_image = tf.reshape(proc_image,[1,32,-1,1]) # Make first dim batch

        with tf.device(FLAGS.device):
            features,sequence_length = model.convnet_layers( proc_image, width, 
                                                             mode)
            logits = model.rnn_layers( features, sequence_length,
                                       mjsynth.num_classes() )
            prediction = _get_output( logits,sequence_length)

        session_config = _get_session_config()
        restore_model = _get_init_trained()
        
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 

        with tf.Session(config=session_config) as sess:
            
            sess.run(init_op)
            restore_model(sess, _get_checkpoint()) # Get latest checkpoint
            # Iterate over filenames given on lines of standard input
            print("[label , predicion , ratio]")
            
            for line in sys.stdin:
                # Eliminate any trailing newline from filename
                ratio = []
                k = 0 
                k = _get_image(k,line.rstrip())
                name = str(line).split(".jpg")[0].split("-")[0]
                for i in range(k):
                    [output] = sess.run(prediction,{ image: all_image[i][1], 
                                                 width: all_image[i][1].shape[1]} )
                    
                    seq = difflib.SequenceMatcher(None, name, _get_string(output.values))
                    ratio.append((i,seq.ratio(),_get_string(output.values)))
                maxprob=max(ratio,key=lambda x:x[1])

                #print(sorted(ratio,key=lambda x:x[1],reverse=True))
                print(name,maxprob[2],round(maxprob[1],2))
                del all_image[:]
				
if __name__ == '__main__':
    all_image = []
    tf.app.run()
