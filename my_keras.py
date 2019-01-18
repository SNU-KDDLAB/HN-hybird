
# coding: utf-8

# In[ ]:

from keras.utils.np_utils import to_categorical
import os, numpy, math
import numpy as np
from random import shuffle
from myLib import Timer
from datetime import datetime

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer
import copy, inspect

    
class My_keras:
    def __init__(self, mini_batch_fn, log, dataset, CHAR, REMAINDER, SEC_PERIOD, n_classes, std_batch_size, 
                 adjust_learning_rate, mse):
        self.mini_batch_fn = mini_batch_fn
        self.log = open(log, 'a', 1)
        self.dataset = dataset
        self.SEC_PERIOD = SEC_PERIOD
        self.n_classes = n_classes
        self.CHAR = CHAR
        self.REMAINDER = REMAINDER
        self.std_batch_size = std_batch_size
        self.adjust_learning_rate = adjust_learning_rate
        self.calc_mse = mse
            
    def __batch_loop(self, train, data_split, learning_rate=None, partial_train=False):
        start_time = datetime.now()
        temp_n, previous_bucket_i, n, n_total, acc_sum, loss_sum, mse_sum, temp_loss_sum, temp_acc_sum                 = 0, None, 0, 0, 0, 0, 0, 0, 0
        
        def print_progress():
            nonlocal temp_n, temp_loss_sum, temp_acc_sum
            if temp_n != 0:
                self.log.write('{:7d} / {:7d}   temp_n={}   temp loss={:.5f}  temp acc={:.5f}  loss={:.5f}  acc={:.5f}'.format(
                        n, previous_n_bucket, temp_n, temp_loss_sum/temp_n, temp_acc_sum/temp_n, loss_sum/n_total, acc_sum/n_total))
            if self.calc_mse:
                  self.log.write('  mse={:.5f}'.format(mse_sum/n_total))
            self.log.write('\n')
            temp_n, temp_loss_sum, temp_acc_sum = 0, 0, 0
        for n_bucket, bucket_i, n_batch, xs, ys, sentence_lengths_list                 in self.dataset.get_array(data_split, self.CHAR):
            # timer process
            if bucket_i!= previous_bucket_i: # when bucket is changed
                timer = Timer(n_bucket, self.SEC_PERIOD)
                n, temp_acc_sum, temp_loss_sum = 0, 0, 0
            previous_bucket_i = bucket_i
            if timer.remaining(n)==True:
                print_progress()
            # learning rate control
            if train and self.adjust_learning_rate:
                K.set_value(self.models[bucket_i].optimizer.lr, learning_rate * n_batch / self.std_batch_size)
            # remainder index process
            bucket_bound = self.dataset.bucket_bounds[bucket_i]
            sentence_remainder = np.zeros((n_batch, bucket_bound[0]), np.float32)
            word_remainder = np.zeros((n_batch, *bucket_bound), np.float32)
            for i, sentence_lengths in enumerate(sentence_lengths_list):
                for j, sentence_length in enumerate(sentence_lengths):
                    word_remainder[i][j][:sentence_length]                         = np.arange(sentence_length - 1 - (sentence_length//2), -1 - (sentence_length//2), -1)
                j += 1
                sentence_remainder[i][:j] = np.arange(j-1- (j//2), -1-(j//2), -1)
            sentence_remainder = numpy.expand_dims(sentence_remainder, -1) / self.dataset.max_sentence_number
            word_remainder = numpy.expand_dims(word_remainder, -1) / self.dataset.max_sentence_length
            
            if type(xs) is not list:
                xs = [xs]
            if self.REMAINDER:
                xs += [sentence_remainder] + [word_remainder]
            n += n_batch
            n_total += n_batch
            temp_n += n_batch
            for i in range(len(xs)):
                xs[i] = xs[i][:n_batch]
            ys = ys[:n_batch]
            ys = to_categorical(ys, self.n_classes)
            if partial_train:  # exclude word embedding information
                xs=xs[1:]
                
            output = self.mini_batch_fn(train, bucket_i, xs, ys)
                
            loss, acc = output[:2]
            if self.calc_mse:
                mse = output[2]
            temp_acc_sum += acc * n_batch
            temp_loss_sum += loss * n_batch
            acc_sum += acc * n_batch
            loss_sum += loss * n_batch
            if self.calc_mse:
                mse_sum += mse * n_batch
            previous_n_bucket = n_bucket
            if n_bucket - n < n_batch:
                print_progress()
        self.log.write('elapsed time: {}\n'.format(datetime.now() - start_time))
        output = [acc_sum / n_total, loss_sum / n_total]
        if self.calc_mse:
            output += [math.sqrt(mse_sum / n_total)]
        else:
            output += [None]
        return output

    def train_model(self, data_split, learning_rate, partial_train=False):
        return self.__batch_loop(True, data_split, learning_rate, partial_train)
    def test_model(self, data_split, partial_train=False):
        return self.__batch_loop(False, data_split, partial_train=partial_train)

