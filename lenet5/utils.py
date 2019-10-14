#coding:utf-8
import os
import tensorflow as tf

def set_config(number, using_rate):
	""" GPU Setting parameters """
	os.environ['CUDA_VISIBLE_DEVICES'] = number
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=using_rate)
	config = tf.ConfigProto(gpu_options=gpu_options)
	return config

def auto_config(number):
	""" Auto GPU Setting """
	os.environ['CUDA_VISIBLE_DEVICES'] = number
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return config
