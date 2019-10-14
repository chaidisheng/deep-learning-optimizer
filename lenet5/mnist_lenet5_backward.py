#coding:utf-8
#cython: language_level = 2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import os
import utils
import numpy as np

# hyperparameter choice
BETA_1 = 0.9
BETA_2 = 0.999 # 0.999
epsilon = 1e-8
EPOCH = 33
BATCH_SIZE = 100
LEARNING_RATE_BASE =  0.005 
LEARNING_RATE_DECAY = 0.99 
REGULARIZER = 0.0001
STEPS = 50000 # 50000 
MOVING_AVERAGE_DECAY = 0.99 
MODEL_SAVE_PATH="./model/" 
MODEL_NAME="mnist_model" 

def backward(mnist):
	x = tf.placeholder(tf.float32,[
	BATCH_SIZE,
	mnist_lenet5_forward.IMAGE_SIZE,
	mnist_lenet5_forward.IMAGE_SIZE,
	mnist_lenet5_forward.NUM_CHANNELS]) 
	y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
	result = mnist_lenet5_forward.forward(x,True, REGULARIZER)

	# train step
	global_step = tf.Variable(0, trainable=False) 
	new_global_step = global_step.assign(global_step + 1)

	# costing function
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result[8], labels=tf.argmax(y_, 1))
	cem = tf.reduce_mean(ce)
	loss = cem + tf.add_n(tf.get_collection('losses')) 
	
	# exp decay learning rate
	learning_rate = tf.train.exponential_decay( 
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples / BATCH_SIZE, 
		LEARNING_RATE_DECAY,
		staircase=True)

	print(mnist.train.num_examples)

	#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	#train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss, global_step=global_step)
	#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

	# compute neural parameters
	for v in tf.trainable_variables():
		print(v)

	print("Neural Parameters are %d." %np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))		

	"""
	# Mini Batch Gradient Descent
	new_result = []
	for res in range(len(result) - 1):
		grad_result = tf.gradients(xs=result[res], ys=loss)
		new_result.append(result[res].assign(result[res] - learning_rate*grad_result[0]))
	"""
	"""
	# Mini Batch Gradient Descent with Momentum
	new_result = []
	v_dw = [0.0 for _ in range(len(result) - 1)]
	for res in range(len(result) - 1):
		grad_result = tf.gradients(xs=result[res], ys=loss)
		v_dw[res] = BETA_1*v_dw[res] + (1 - BETA_1)*grad_result[0]
		new_result.append(result[res].assign(result[res] - learning_rate*v_dw[res]))
	"""
	
	"""
	# RMSPROP
	new_result = []
	s_dw = [0 for _ in range(len(result) - 1)]
	for res in range(len(result) - 1):
		gradient = tf.gradients(xs=result[res], ys=loss)
		s_dw[res] = BETA_2*s_dw[res] + (1 - BETA_2)*tf.square(gradient[0])
		new_result.append(result[res].assign(result[res] - learning_rate*gradient[0] / (tf.sqrt(s_dw[res]) + epsilon)))
	"""

	# Adam
	new_result = []
	v_dw = [0.0 for _ in range(len(result) - 1)]
	s_dw = [0.0 for _ in range(len(result) - 1)]
	for res in range(len(result) - 1):
		gradient = tf.gradients(xs=result[res], ys=loss)
		v_dw[res] = BETA_1*v_dw[res] + (1 - BETA_1)*gradient[0]
		s_dw[res] = BETA_2*s_dw[res] + (1 - BETA_2)*tf.square(gradient[0])
		v_dw[res] = v_dw[res] / (1 - tf.pow(BETA_1, tf.cast(global_step, dtype=tf.float32)))
		s_dw[res] = s_dw[res] / (1 - tf.pow(BETA_2, tf.cast(global_step, dtype=tf.float32)))
		new_result.append(result[res].assign(result[res] - learning_rate*v_dw[res] / (tf.sqrt(s_dw[res] + epsilon))))


	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	ema_op = ema.apply(tf.trainable_variables())

	# Binding node training together
	# with tf.control_dependencies([train-step, ema_op]):
	#	train_op = tf.no_op(name='train')

	saver = tf.train.Saver() # max_keep_dim = 3 
	
	with tf.Session(config=utils.auto_config('1, 0')) as sess: 
		init_op = tf.global_variables_initializer() 
		sess.run(init_op) 

		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) 
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path) 
		for epoch in range(EPOCH):
			for i in range(STEPS):
				xs, ys = mnist.train.next_batch(BATCH_SIZE) 
				reshaped_xs = np.reshape(xs,( 
					BATCH_SIZE,
					mnist_lenet5_forward.IMAGE_SIZE,
					mnist_lenet5_forward.IMAGE_SIZE,
					mnist_lenet5_forward.NUM_CHANNELS))
			
				# new_result[:] : can't using
				_, _, loss_value, step = sess.run([ema_op, new_result, loss, new_global_step], feed_dict={x: reshaped_xs, y_: ys}) 
				if not i % 100: 
					print("Epoch %03d/%03d | Batch %d/%d, loss on training batch is %g." % (epoch + 1, EPOCH, step, STEPS, loss_value))
					saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
	mnist = input_data.read_data_sets("./data/", one_hot=True) 
	backward(mnist)

if __name__ == '__main__':
	main()


