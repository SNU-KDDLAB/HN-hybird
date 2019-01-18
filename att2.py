
# coding: utf-8

# In[1]:

DIMENSION_ATTENTION = 100
import numpy as np
import os, json, pickle, re, sys, math, socket
from etc import Dataset
from path import Path
from my_keras import My_keras
from saver import find_new_dir, save_src
from numpy import argmax
from pprint import pprint
import shutil

import keras
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.wrappers import Bidirectional, TimeDistributed, Wrapper
from keras.layers.core import Dropout, Dense, Lambda, Masking, Flatten
from keras.layers.merge import Multiply
from keras.engine.topology import Layer
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.layers.merge import Concatenate, concatenate, Maximum, multiply, add
from keras.initializers import Orthogonal, Constant

from keras import backend as K
from keras import initializers, optimizers, regularizers
from keras.regularizers import l2
from keras.utils.generic_utils import CustomObjectScope

from random import randrange
# import theano


# In[3]:

if False:
    dataset.calc_word_count()
    dense_word_cnt = 0
    sparse_word_cnt = 0
    sys.stdout = open('sparse_words.txt', 'w', 1)
    for x in dataset.sorted_word_count:
        word = x[0]
        word_cnt = x[1]
        if word_cnt<5:
            sparse_word_cnt += word_cnt
            print(word, '\t', word_cnt)
        else:
            dense_word_cnt += word_cnt
    print(dense_word_cnt, sparse_word_cnt)


# In[4]:

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.Uw = self.add_weight(shape=(input_dim, 1), initializer='glorot_uniform', trainable=True, name='att_W')
        super(AttentionLayer, self).build(input_shape)  
    
    def compute_mask(self, input, mask):
        return mask
    
    def call(self, x, mask=None):
        multData =  K.exp(K.dot(x, self.Uw))    # (.. , 1)
        if mask is not None:
            masked_att = multData * K.expand_dims(K.cast(mask, dtype='float32'))
        else:
            masked_att = multData
        att = masked_att / (K.sum(masked_att, axis=1, keepdims=True) + K.epsilon())
        att = K.repeat_elements(att, DIMENSION_ATTENTION, 2)
        return att

    def compute_output_shape(self, input_shape):
        return (input_shape[0], DIMENSION_ATTENTION)
    
class Switch(Layer):
    def __init__(self, dim, n_dim, **kwargs):
        super(Switch, self).__init__(**kwargs)
        self.dim = dim
        self.n_dim = n_dim
        self.supports_masking = True
        
    def build(self, input_shape):
        super(Switch, self).build(input_shape)  

    def call(self, x): # x[0], x[1]: (batch_size, n)   x[2]: (batch size, n, dim)
        # x[0]: exists array   x[1]: known array
        assert len(x)==self.n_dim
        exists = K.expand_dims(x[0])
        exists = K.repeat_elements(exists, self.dim, axis=self.n_dim-1)
        known = K.expand_dims(x[1])
        known = K.repeat_elements(known, self.dim, axis=self.n_dim-1)
        if True:
            switched = (known * x[2]) + (1-known * x[3])
            return switched * exists
        else:
            if K.backend() == 'tensorflow':
                switch = tf.cond
                transpose = tf.transpose
            elif K.backend() == 'theano':
                switch = theano.ifelse.ifelse
                transpose = theano.tensor.transpose
            stacked = K.stack([exists, known, x[2], x[3]])
            transposed = transpose(stacked, [1,2,3,4,0])
            def foo(x):
                exist = x[0]
                known = x[1]
                switched = switch(known, x[2], x[3])
                return switch(exist, switched, 0)
            switched = K.map_fn(foo, transposed)
            return switched
        
#         switched = add([multiply([known, x[2]]), multiply([1-known, x[3]])])
#         masked = multiply([switched, exists])
#         return masked
    def compute_output_shape(self, input_shape):
        return input_shape[2]
def argmax_mse(y_true, y_pred):
    return K.cast(K.square(K.argmax(y_true, axis=-1) - K.argmax(y_pred, axis=-1)), K.floatx())


# In[ ]:

class Multi(Wrapper):
# embedded = TimeDistributed(self.word_embedding_layer)(wordsInputs)
    def __init__(self, emb_dim, layer, **kwargs):
        super(Multi, self).__init__(layer, **kwargs)
        self.supports_masking = True
        self.emb_dim = emb_dim
        self.layer = layer
    def build(self, input_shape):
        super(Multi, self).build(input_shape)  
    def call(self, x, emb_layer): # x: (n, ...)
        batch_size = K.int_shape(x)[0]
        out = []
        for i in range(batch_size):
            out += []
    def compute_output_shape(self, input_shape):
        return (*input_shape, self.emb_dim)


# In[ ]:

class MyFlat(Layer):
    def __init__(self, **kwargs):
        super(MyFlat, self).__init__(**kwargs)
    def build(self, input_shape):
        super(MyFlat, self).build(input_shape)  
    def call(self, x): # x: (?, m, n)   out: (?, m*n)
        input_shape = K.int_shape(x)
        assert len(input_shape) == 3
        return K.batch_flatten(x)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[2])


# In[5]:

'''
if there is save file in save_dir, load the process, else create new numbered directory.
data_directory: only used in 'path' module
'''
class HN():
    def __init__(self, src_list, save_dir, data_directory, data_name, glove, word_coverage, non_test,
                 initial_learning_rate, learning_rate_decay, optimizer_kwargs, adjust_learning_rate, clip_batch_size,
                 set_min_batch_size, char_rnn, REMAINDER, CHAR, MAX_WORD_LENGTH, BUCKET_DIVISION,
                 WordEmb_dropout, WordRnn_dropout, SentenceRnn_dropout, rnn,
                 CHARACTER_RNN_DIMENSION, conv_unit_size,
                 mse, TRAIN, pretrain, 
                 mode,
                 mem_test=False, test=False, bucket_coverage=None,
                 max_n_epoch = 55, cnn_window_size=7,
                 memory=False, optimizer_name='rmsprop', std_batch_size=64,
                 embedding_regularizer_coefficient=None, 
                 recurrent_activation='sigmoid', conv_dense_activation='tanh',
                entire_char_size=50, kernel_regularizer_coefficient=None, sec_period=30, memory_fraction=1,
                 stack=[1, 1], RNN_DIMENSION=50, trainable_word_emb=True, year=None, 
                 flag_embedding_layer_mask_zero=True, rnn_implementation=2,
                 tanh2_dropout=0, patience=3, complete_pretrain=False
                ):
        save_filename = 'model.h5'
        if os.path.isfile(os.path.join(save_dir, save_filename)):
            print('===== will load weights ======')
        else:  # create new path
            save_dir = find_new_dir(save_dir)
        # copy source codes
        save_src(save_dir, src_list)
        # open log file
        log_path = os.path.join(save_dir, 'log.txt')
        self.log, sys.stdout = open(log_path, 'a', 1), open(log_path, 'a', 1)
        self.log.write('\n\n\n\n\n')
        # adjust config
        if test:
            data_name = 'small'
            max_n_epoch = 3
            BUCKET_DIVISION = 2
            conv_unit_size = 32
        if set_min_batch_size and K.backend()=='theano':
            adjust_learning_rate = False
            import theano
            theano.config.optimizer = 'fast_run'
        if pretrain:
            print('pre-train is impossible now due to fast-mem_test')
            sys.exit()
        # parameters
        self.log_path, self.sec_period, self.std_batch_size = log_path, sec_period, std_batch_size
        self.adjust_learning_rate = adjust_learning_rate
        self.best_validation_save_path = os.path.join(save_dir, 'best-accuracy.txt')
        self.best_save_path = os.path.join(save_dir, 'best.h5')
        self.model_save_path = os.path.join(save_dir, save_filename)
        self.mem_test_flag = mem_test
        self.learning_rate_decay = learning_rate_decay
        self.custom_objects = {'AttentionLayer':AttentionLayer}
        self.word_embedding_dim = 200
        self.rnn_implementation = rnn_implementation
        self.trainable_word_emb, self.flag_embedding_layer_mask_zero = trainable_word_emb, flag_embedding_layer_mask_zero
        self.CHAR = CHAR
        self.REMAINDER = REMAINDER
        self.rnn, self.conv_dense_activation = rnn, conv_dense_activation
        self.recurrent_activation = recurrent_activation
        self.initial_learning_rate = initial_learning_rate
        self.TRAIN = TRAIN
        self.stack = stack
        self.pretrain, self.complete_pretrain = pretrain, complete_pretrain
        self.patience, self.mse, self.max_n_epoch = patience, mse, max_n_epoch
        self.MULTI_RNN_DIMENSION = RNN_DIMENSION  # int(RNN_DIMENSION / math.sqrt(N_LAYERS))
        self.conv_unit_size = conv_unit_size
        self.embedding_regularizer_coefficient = embedding_regularizer_coefficient
        self.kernel_regularizer_coefficient = kernel_regularizer_coefficient
        self.char_rnn_flag, self.CHARACTER_RNN_DIMENSION = char_rnn, CHARACTER_RNN_DIMENSION
        self.WordEmb_dropout, self.WordRnn_dropout, self.SentenceRnn_dropout, self.tanh2_dropout                 = WordEmb_dropout, WordRnn_dropout, SentenceRnn_dropout, tanh2_dropout
        self.optimizer_name, self.optimizer_kwargs = optimizer_name, optimizer_kwargs
        self.save_dir = save_dir
        self.mode = mode
        # prepare dataset     
        self.path = Path(directory_path=data_directory, name=data_name, glove=glove, year=year)
        MEMORY = self.memory_control(memory_fraction)
        if memory:
            MEMORY = memory
        self.log.write('MEMORY: {}\n'.format(MEMORY))
        self.dataset = Dataset(log_path, self.path, set_min_batch_size=set_min_batch_size,
                bucket_coverage=bucket_coverage, clip_batch_size=clip_batch_size,
               non_test_flag=non_test, word_coverage=word_coverage, save_dir=save_dir, 
               MAX_WORD_LENGTH=MAX_WORD_LENGTH, MAX_N_CHARACTER=entire_char_size)
        self.dataset.bucketize2(BUCKET_DIVISION, MEMORY)
        self.dataset.import_word_embedding(EMBEDDING_DIM=200)
        self.log.write('vocabulary_size={}, embedding_matrix_size, EMBEDDING_DIM = {}, {}\n'.format(
                len(self.dataset.word_count), len(self.dataset.word_embedding_matrix), self.word_embedding_dim))
        self.dataset.std_batch_size = std_batch_size
        self.max_word_length = self.dataset.MAX_WORD_LENGTH
        # print configurations
        self.log.write('test={}   mem_test={}   memory_fraction = {}\n'.format(test, mem_test, memory_fraction))
        self.log.write('data_name = {} {}\n'.format(data_name, year))
        self.log.write('set_min_batch_size = {}   bucket_coverage = {}\n'.format(set_min_batch_size, bucket_coverage))
        self.log.write('TRAIN = {}   mse = {}\n'.format(TRAIN, mse))
        self.log.write('mode = {}'.format(self.mode))
        self.log.write('glove = {},   non_test = {}   word_coverage = {}\n'.format(
            glove, non_test, word_coverage))
        self.log.write('entire_char_size={},  MAX_WORD_LENGTH = {},  BUCKET_DIVISION = {}\n'.format(
            entire_char_size, MAX_WORD_LENGTH, BUCKET_DIVISION))
        self.log.write('save_dir = {}\n'.format(save_dir))
        self.log.write('log_path = {}\n'.format(log_path))
        self.log.write('optimizer_kwargs = {}   initial_learning_rate = {}\n'.format(
            optimizer_kwargs, initial_learning_rate))
        self.log.write('learning_rate_decay = {}\n'.format(learning_rate_decay))
        self.log.write('backend: {}\n'.format(K.backend()))
        self.log.write('REMAINDER = {}\n'.format(REMAINDER))
        self.log.write('CHAR = {},  char_rnn = {}\n'.format(CHAR, char_rnn))
        self.log.write('embedding_regularizer_coefficient = {}\n'.format(embedding_regularizer_coefficient))
        self.log.write('kernel_regularizer_coefficient = {}\n'.format(kernel_regularizer_coefficient))
        self.log.write('rnn = {},  stack = {}  conv_unit_size={}\n'.format(rnn, stack, conv_unit_size))
        self.log.write('optimizer_name = {}\n'.format(optimizer_name))
        self.log.write('cnn_window_size = {}\n'.format(cnn_window_size))
        self.log.write('CHARACTER_RNN_DIMENSION = {}\n'.format(CHARACTER_RNN_DIMENSION))
        self.log.write('std_batch_size = {}\n'.format(std_batch_size))
        self.log.write('WordEmb_dropout, WordRnn_dropout, SentenceRnn_dropout = {}, {}, {}\n'.format(
                WordEmb_dropout, WordRnn_dropout, SentenceRnn_dropout))
        self.log.write('pretrain = {}   trainable_word_emb = {}\n'.format(pretrain, trainable_word_emb))
        self.log.write('adjust_learning_rate = {}\n'.format(adjust_learning_rate))
        self.log.write('tanh2_dropout = {}\n'.format(tanh2_dropout))
        self.log.write('patience ={}\n'.format(patience))
        self.log.write('clip_batch_size = {}\n'.format(clip_batch_size))
        self.log.write('conv_dense_activation = {}    recurrent_activation = {}\n'.format(conv_dense_activation, recurrent_activation))
        self.make_layers()

    def run(self):
        if self.mem_test_flag is True:
            self.mem_test()
            self.run_without_mem_test(compiled=True)
        else:
            self.run_without_mem_test()
        print('=== finished ===')
    def mem_test(self):
        self.dataset.mem_test, self.mem_test_flag = True, True
        print('=== memory test ====')
        self.run_without_mem_test()
        self.dataset.mem_test, self.mem_test_flag = False, False
    def make_layers(self):
        self.word_embedding_layer = Embedding(len(self.dataset.word_embedding_matrix), 
                                              self.word_embedding_dim,
                embeddings_initializer=Constant(self.dataset.word_embedding_matrix),
                trainable=self.trainable_word_emb,
                mask_zero=self.flag_embedding_layer_mask_zero,
                 name = 'word_emb', embeddings_regularizer=l2(self.embedding_regularizer_coefficient))
        self.word_rnn = self.stack_rnn(self.stack[0])
        self.tanh1 = self.make_dense(DIMENSION_ATTENTION, activation='tanh')
        self.att1 = AttentionLayer(name='att1')
        
        self.rnn2 = self.stack_rnn(self.stack[1])
        self.tanh2 = self.make_dense(DIMENSION_ATTENTION, activation='tanh')
        self.att2 = AttentionLayer(name='att2')
        self.logit = self.make_dense(self.dataset.n_classes, activation='softmax', name='logit')
        if self.optimizer_name == 'rmsprop':
            self.optimizer = optimizers.RMSprop(lr=self.initial_learning_rate, **self.optimizer_kwargs)
        elif self.optimizer_name == 'sgd':
            self.optimizer = optimizers.SGD(lr=self.initial_learning_rate, momentum=0.9, **self.optimizer_kwargs)
        elif self.optimizer_name == 'adam':
            self.optimizer = optimizers.Adam(lr=self.initial_learning_rate, **self.optimizer_kwargs)
        else:
            self.log.write('unknown optimizer name')
            sys.exit(1)
        if self.CHAR:
            self.dataset.make_character_embedding_index()
            self.character_embedding_layer = Embedding(len(self.dataset.char_embedding_index),
                                                       len(self.dataset.char_embedding_index)-2, 
                                                       # weights=[self.dataset.char_embedding_matrix], 
                        embeddings_initializer=Constant(self.dataset.char_embedding_matrix),                                                        
                       mask_zero=self.char_rnn_flag, trainable=True, name='ch_emb',
                      embeddings_regularizer=l2(self.embedding_regularizer_coefficient))
            if not self.char_rnn_flag: # character cnn
                self.conv1 = self.make_conv(self.conv_unit_size, 5)
                self.conv2 = self.make_conv(self.conv_unit_size, 2)
            else: # character rnn
                self.char_rnn = self.make_rnn(self.CHARACTER_RNN_DIMENSION, False)
            # temp
            self.word_linear = self.make_dense(self.word_embedding_dim, activation='linear')
            self.char_linear = self.make_dense(self.word_embedding_dim, activation='linear')
            self.max_tanh = self.make_dense(self.word_embedding_dim, activation='tanh')
            # layers for merging words and characters
            self.conv_dense = self.make_dense(self.word_embedding_dim, activation=self.conv_dense_activation)
            self.max_relu = self.make_dense(self.word_embedding_dim, activation='relu')
    def memory_control(self, memory_fraction):
        memory = {'citron':11169, 'apple':8108, 'cacao':4041, 'lime':3300, 'tangerine':4036} # 'lime':3050
        # memory['citron'] = memory['cacao']
        # memory['apple'] = memory['cacao']
        # memory['tangerine'] = memory['cacao']
        # memory['lime'] = memory['cacao']
        memory['durian'] = memory['citron'] # 11169
        memory['lemon'] = memory['apple']
        host = socket.gethostname()
        if self.CHAR:
            MEMORY = int(10 * memory[host] * memory_fraction * 0.8)
            if False and memory[host]>11000 and memory_fraction > 0.85:
                MEMORY = int(0.8 * MEMORY)
        else:
            MEMORY = int(10 * memory[host] * memory_fraction * 0.8)
        if K.backend()=='tensorflow' and memory_fraction < 0.85:
            import tensorflow as tf
            from keras.backend.tensorflow_backend import set_session
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
            set_session(tf.Session(config=tf_config))
        return MEMORY
    def make_conv(self, unit_size, kernel_size):
        return Conv1D(unit_size, kernel_size, activation='relu',
                      kernel_regularizer=l2(self.kernel_regularizer_coefficient))
    def stack_rnn(self, stack):
        rnn = []
        for _ in range(stack):
            rnn += [self.make_bi_rnn()]
        return rnn
    
    def make_bi_rnn(self):
        return Bidirectional(self.make_rnn(self.MULTI_RNN_DIMENSION, True))

    def make_rnn(self, dimension, return_sequences=False):
        if self.rnn=='gru':
            rnn = GRU
        elif self.rnn=='lstm':
            rnn = LSTM
        return rnn(dimension, return_sequences=return_sequences,
                    implementation=self.rnn_implementation,
                   recurrent_activation = self.recurrent_activation,
                  kernel_regularizer=l2(self.kernel_regularizer_coefficient), 
                   recurrent_regularizer=l2(self.kernel_regularizer_coefficient))

    def make_dense(self, dim, activation, name=None):
        return Dense(dim, activation=activation, name=name,
                     kernel_regularizer=l2(self.kernel_regularizer_coefficient))
    
    def char_to_word_model(self, max_word_length):  # char_i_input: (batch_size, word-length)
        char_i_input = Input(shape=(max_word_length,), dtype='int32', name='word_ch_input')
        embedded_characters = self.character_embedding_layer(char_i_input)
        if not self.char_rnn_flag:
            conv_tensor = self.conv1(embedded_characters)
            conv_tensor = MaxPooling1D(3)(conv_tensor)
            conv_tensor = self.conv2(conv_tensor)
            conv_tensor = MaxPooling1D(2)(conv_tensor)
            # conv_out_shape = K.int_shape(conv_tensor)
            # output = Flatten()(conv_tensor)
            # flatt_out_shape = (conv_out_shape[0], conv_out_shape[1]*conv_out_shape[2])
            # output = Lambda(lambda x: Flatten()(x), output_shape=flatt_out_shape)(conv_tensor)
            output = MyFlat()(conv_tensor)
        else:
            output = self.char_rnn(embedded_characters)
        print('flatten: ', output)
        model = Model(char_i_input, output)
        print('model.output_shape=', model.output_shape)
        return model

    def embedded_word_to_sentence(self, max_sentence_length, embed_dim):
        embedded_word = Input(shape=(max_sentence_length, embed_dim), dtype='float32', name='emb_word_in')
        masked = Masking()(embedded_word)
        masked = Dropout(self.WordEmb_dropout)(masked)
        for i in range(self.stack[0]):
            masked = self.word_rnn[i](masked)
        wordRnn = Dropout(self.WordRnn_dropout)(masked)
        word_tanh = self.tanh1(wordRnn)
        # word_tanh = Dropout(self.WordRnn_dropout)(word_tanh)
        attention = self.att1(word_tanh)
        sentenceEmb = Multiply()([wordRnn, attention])
        sentenceEmb = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0],x[2]))(sentenceEmb)
        modelSentence = Model(embedded_word, sentenceEmb)
#         print('word_to_sentence model summary')
#         print(modelSentence.summary())
        # modelSentAttention = Model(embedded_word, attention)
        return modelSentence
    
    def embed_word_document(self, wordsInputs):
        embedded = TimeDistributed(self.word_embedding_layer)(wordsInputs)
        return Masking()(embedded)
    
    def embed_char_document(self, char_input):
        # char_input: (batch_size, n_sentences, sentence_length, word_length)
        if False:
            model = self.char_to_word_model(self.max_word_length)  # char_i_input: (batch_size, word-length)
            return TimeDistributed(TimeDistributed(model))(char_input)
        else:
            embedded = TimeDistributed(TimeDistributed(self.character_embedding_layer))(char_input)
            conv_tensor = TimeDistributed(TimeDistributed(self.conv1))(embedded)
            conv_tensor = TimeDistributed(TimeDistributed(MaxPooling1D(3)))(conv_tensor)
            conv_tensor = TimeDistributed(TimeDistributed(self.conv2))(conv_tensor)
            conv_tensor = TimeDistributed(TimeDistributed(MaxPooling1D(2)))(conv_tensor)
            output = TimeDistributed(TimeDistributed(Flatten()))(conv_tensor)
            output = self.conv_dense(output)
            return output

    def embedded_word_to_document(self, max_sentence_length, embed_dim, embedded, sentence_remainder=None, word_remainder=None): 
        # input:(batch_size, sentence_size, sentence_length, embed_dim)
        # assume input is masked
        # embedded = append_word_remainder(word_remainder, words)
        # sentence level
        if self.REMAINDER:
            # expanded_word_remainder = K.expand_dims(word_remainder, axis=-1)
            embedded = Concatenate()([word_remainder, embedded])
            embed_dim += 1
        modelSentence = self.embedded_word_to_sentence(max_sentence_length, embed_dim)
        sentenceEmbbeding = TimeDistributed(modelSentence)(embedded)
        # sentenceAttention = TimeDistributed(modelSentAttention)(embedded)
        # document level
        sentenceEmbbeding = Masking()(sentenceEmbbeding)
        if self.REMAINDER:
            # expanded_sentence_remainder = K.expand_dims(sentence_remainder, axis=-1)
            sentenceEmbbeding = Concatenate()([sentence_remainder, sentenceEmbbeding])
        for i in range(self.stack[1]):
            sentenceEmbbeding = self.rnn2[i](sentenceEmbbeding)
        # sentenceEmbbeding = Dropout(self.SentenceRnn_dropout)(sentenceEmbbeding)
        sentence_tanh = self.tanh2(sentenceEmbbeding)
        sentence_tanh = Dropout(self.tanh2_dropout)(sentence_tanh)
        attentionSent = self.att2(sentence_tanh)
        documentEmb = Multiply()([sentenceEmbbeding, attentionSent])
        documentEmb = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0],x[2]), name="sum_att2")(documentEmb)
        documentOut = self.logit(documentEmb)
        return documentOut
    
    def append_word_remainder(self, word_remainder, words):
        if self.REMAINDER:
            words = concatenate([word_remainder, words])
        return words
    
    def remainder_input_tensor(self, max_n_sentences, max_length):
        if self.REMAINDER:
            # sentence and word remainder
            return Input(shape=(max_n_sentences, 1), dtype='float32'),                     Input(shape=(max_n_sentences, max_length, 1), dtype='float32')
        else:
            return None, None    
    def word_model(self, batch_size, max_n_sentences, max_length):
        # embed input
        sentence_remainder, word_remainder = self.remainder_input_tensor(max_n_sentences, max_length)
        documentInputs = Input(shape=(max_n_sentences, max_length), dtype='int32', name='word_input')
        embedded = self.embed_word_document(documentInputs) # masked
        output = self.embedded_word_to_document(
            max_length, self.word_embedding_dim, embedded, sentence_remainder, word_remainder)
        # model creation
        if not self.REMAINDER:
            model = Model([documentInputs], output)
        else:
            model = Model([documentInputs, sentence_remainder, word_remainder], output)
        # modelAttentionEv = Model(inputs=[documentInputs], outputs=[output,  sentenceAttention, attentionSent])
        self.compile_model(model)
        # self.compile_model(modelAttentionEv)
        return model
    
    def combined_model(self, batch_size, max_n_sentences, max_sentence_length, max_word_length):
        sentence_remainder, word_remainder = self.remainder_input_tensor(max_n_sentences, max_sentence_length)
        exists = Input(shape=(max_n_sentences, max_sentence_length), dtype='float32',
                       name='exists')
        known = Input(shape=(max_n_sentences, max_sentence_length), dtype='float32',
                      name='known')
        # embed word
        word_input = Input(shape=(max_n_sentences, max_sentence_length), dtype='int32', name='doc-wd_in')
        embedded_word = self.embed_word_document(word_input) # masked
        # embed char
        char_input = Input(shape=(max_n_sentences, max_sentence_length, max_word_length), dtype='int32', name='doc_ch_in')
        embedded_char = self.embed_char_document(char_input)
        # combine (masked during merging)
        if False:
            # concat_word = Concatenate()([embedded_char, embedded_word]) 
            # concat_word = self.conv_dense(concat_word)
            embedded_word = self.max_relu(embedded_word)
            embedded_char = self.max_relu(embedded_char)
            concat_word = Maximum()([embedded_word, embedded_char])
            # concat_word = self.max_tanh(concat_word)
        elif self.mode == 'max':
            # embedded_word = self.word_linear(embedded_word)
            # embedded_char = self.char_linear(embedded_char)
            concat_word = Maximum()([embedded_word, embedded_char])
            # concat_word = self.max_tanh(concat_word)
        elif self.mode == 'switch':
            if True:
                concat_word = Switch(self.word_embedding_dim, 4)([exists, known, embedded_word, embedded_char])
            else:
                concat_word = embedded_char
        def expand(x):
            y = K.expand_dims(exists)
            return K.repeat_elements(y, self.word_embedding_dim, axis=3)
        def calc_output_shape(in_shape):
            return in_shape + (self.word_embedding_dim,)
        expanded_exists = Lambda(expand, output_shape=calc_output_shape)(exists)
        concat_word = Multiply()([concat_word, expanded_exists])
        # to document
        output = self.embedded_word_to_document(
            max_sentence_length, self.word_embedding_dim, concat_word, sentence_remainder, word_remainder)
        if not self.REMAINDER:
            model = Model([exists, known, word_input, char_input], output)
        else:
            model = Model([exists, known, word_input, char_input, sentence_remainder, word_remainder], output)
        self.compile_model(model)
        return model
    
    def compile_model(self, model):
        metrics = ['accuracy']
        if self.mse:
            metrics += [argmax_mse]
        model.compile(loss='categorical_crossentropy',
          optimizer=self.optimizer, metrics=metrics)
        
    def make_models(self):
        self.models = []
        if self.CHAR:
            compile_models = self.bucket_combined_models
        else:
            compile_models = self.bucket_word_models
        compile_models()
        self.log.write(str(self.models[0].summary()) + '\n')
        
        self.make_mini_batch_fn()
        
        self.my_keras = My_keras(self.mini_batch_fn, self.log_path, self.dataset, self.CHAR, self.REMAINDER, 
                                 self.sec_period, self.dataset.n_classes, self.std_batch_size, 
                                 adjust_learning_rate=self.adjust_learning_rate, mse=self.mse)
        return self.models
    
    def make_mini_batch_fn(self):
        def mini_batch_fn(is_train, bucket_i, xs, ys):
            if is_train:
                return self.models[bucket_i].train_on_batch(xs, ys)
            else:
                return self.models[bucket_i].test_on_batch(xs, ys)
        self.mini_batch_fn = mini_batch_fn
    
    def bucket_word_models(self):
        for i in range(len(self.dataset.bucket_bounds)):
            batch_size = self.dataset.bucket_batch_size[i]
            max_n_sentences = self.dataset.bucket_bounds[i][0]
            max_sentence_length = self.dataset.bucket_bounds[i][1]
            self.models += [self.word_model(batch_size, max_n_sentences, max_sentence_length)]
    def bucket_combined_models(self):
        for i in range(len(self.dataset.bucket_bounds)):
            batch_size = self.dataset.bucket_batch_size[i]
            max_n_sentences = self.dataset.bucket_bounds[i][0]
            max_sentence_length = self.dataset.bucket_bounds[i][1]
            self.models += [self.combined_model(batch_size,
                max_n_sentences, max_sentence_length, self.dataset.MAX_WORD_LENGTH)]    
    def set_learning_rate(self, models, learning_rate):
        K.set_value(self.optimizer.lr, learning_rate)

    def run_epochs(self, models, save_path, max_n_epoch, pretrain=False, decay=None, 
                   partial_train=False, best_validation_accuracy=0):
        self.my_keras.models = models
        best_val_acc, n_patient = best_validation_accuracy, 0
        learning_rate = self.initial_learning_rate
        if not decay:
            decay = self.learning_rate_decay
        for i in range(max_n_epoch):
            self.log.write('learning_rate = {}\n'.format(learning_rate))
            if self.TRAIN:
                self.log.write('==== train ==== (epoch {})\n'.format(i+1))
                train_acc, train_loss, train_rmse = self.my_keras.train_model(
                        self.dataset.train, learning_rate, partial_train=partial_train)
            self.log.write('========== validation ==========\n')
            val_acc, val_loss, val_rmse = self.my_keras.test_model(self.dataset.validation, partial_train=partial_train)
            self.log.write('========== test ==========\n')
            test_acc, test_loss, test_rmse = self.my_keras.test_model(self.dataset.test, partial_train=partial_train)
            self.log.write('{}   '.format(i+1))
            if self.TRAIN:
                self.log.write('train loss = {:.5f}  train rmse = {}'.format(train_loss, train_rmse))
            self.log.write('   val_loss = {:.5f}  val_rmse={}'.format(val_loss, val_rmse))
            self.log.write('   test_loss = {:.5f}  test_rmse={}\n'.format(test_loss, test_rmse))
            if self.TRAIN:
                self.log.write('train_acc = {:.5f}  '.format(train_acc))
            self.log.write('val_acc = {:.5f}  test_acc = {:.5f}\n'.format(val_acc, test_acc))
            with open('acc.txt', 'a') as f:
                f.write('val_loss = {:.5f},  val_acc = {:.5f},  test_loss = {:.5f},  test accuracy = {:.5f}\n'.format(
                    val_loss, val_acc, test_loss, test_acc))
            self.print_emb_matrix()
            # save the current
            if not self.mem_test_flag:
                models[0].save_weights(save_path)
            # save the best
            prev_best_val_acc = best_val_acc
            if not self.mem_test_flag and val_acc >= best_val_acc + 0.0005:
                best_val_acc = val_acc
                models[0].save_weights(self.best_save_path)
                self.save_best_validation(best_val_acc)
                self.log.write('====== saved ======\n')
            # increace patience if the improvement is not enough
            if val_acc >= prev_best_val_acc + 0.0005:
                n_patient = 0
            else:
                n_patient += 1
                print('n_patient = {}'.format(n_patient))
                if n_patient >= self.patience:
                    break
            # the break condition for pretrain
            if pretrain and val_acc - best_val_acc < 0.01:
                break # terminate when pretrain and small difference
                
            learning_rate /= decay
            self.set_learning_rate(models, learning_rate)
            
            if self.mem_test_flag:
                break
            
    def load(self, models, save_path):
        if os.path.exists(save_path):
            with CustomObjectScope(self.custom_objects):
                models[0].load_weights(save_path)
                with open(self.best_validation_save_path, 'r') as f:
                    best_val_acc = float(f.read())    # error if there is no float value in the file
            self.log.write('====== {} loaded (best validation accuracy = {}) =====\n'.format(save_path, best_val_acc))
            return best_val_acc
        else:
            self.log.write('======== failed loading .... NEW   {} =========\n'.format(save_path))
            self.print_emb_matrix('before save')
            models[0].save_weights(save_path) # save initial restore point for mem_test
            self.print_emb_matrix('after save')
            self.save_best_validation(0)
            return 0
    def run_without_mem_test(self, compiled=False):
        if not self.CHAR:  # only word
            self.log.write('========== word-model ==========\n')
            if not compiled:
                self.make_models()
            best_validation_accuracy = self.load(self.models, self.model_save_path)
            self.run_epochs(self.models, self.model_save_path, self.max_n_epoch, best_validation_accuracy=best_validation_accuracy)
        else:
            self.log.write('=== combined model ======\n')
            concat_save_path = os.path.join(self.save_dir, 'concat.h5')
            if not self.mem_test and self.pretrain and not os.path.exists(concat_save_path):
                # pre-train
                self.word_embedding_layer.trainable = False
                pretrain_models = self.make_models()
                pretrain_save_path = os.path.join(self.save_dir, 'pretrain.h5')
                if not self.complete_pretrain or not os.path.exists(pretrain_save_path):
                    self.log.write('=== pretrain ====\n')
                    best_validation_accuracy = self.load(pretrain_models, self.best_save_path)
                    self.run_epochs(pretrain_models, pretrain_save_path, max_n_epoch=2, decay=1,
                                    best_validation_accuracy=best_validation_accuracy)
                else:
                    self.log.write('=== finish pre-train ====\n')
                # convert to full-model
                self.load(pretrain_models, self.best_save_path)
                self.word_embedding_layer.trainable = True
                concat_models = self.make_models()
                concat_models[0].save_weights(concat_save_path)
                self.clear()
            if not compiled:
                self.make_models()
            best_validation_accuracy = self.load(self.models, self.model_save_path)
            self.run_epochs(self.models, self.model_save_path, self.max_n_epoch, best_validation_accuracy=best_validation_accuracy)
    def save_best_validation(self, accuracy):
        with open(self.best_validation_save_path, 'w') as f:
            f.write(str(accuracy))
    def clear(self):
        print('cleared')
        if os.environ['KERAS_BACKEND']=='tensorflow':
            K.clear_session()
            self.make_layers()
    def print_emb_matrix(self, in_str=None):
        with open(os.path.join(self.save_dir, 'weights.txt'), 'a') as f:
            f.write('=======================\n')
            if in_str:
                f.write(in_str + '\n')
            f.write('word_rnn = {}\n'.format(self.word_rnn[0].get_weights()))
            f.write('word = {}\n'.format(self.word_embedding_layer.get_weights()))
            if self.CHAR:
                f.write('char = {}\n'.format(self.character_embedding_layer.get_weights()))
                f.write('conv1 = {}\n'.format(self.conv1.get_weights()))
                f.write('conv2 = {}\n'.format(self.conv2.get_weights()))


# In[7]:

def make_model():
    return HN(hn.WordEmb_dropout, hn.WordRnn_dropout, hn.SentenceRnn_dropout,
               embedding_regularizer_coefficient=config.embedding_regularizer_coefficient,
               kernel_regularizer_coefficient=config.kernel_regularizer_coefficient,
               rnn=config.rnn)


if False:# else: # contains char model
    concat_save_path = 'concat.h5'
    best_char_path = 'best_char.h5'
    if not os.path.exists(best_char_path) and not config.SKIP_PRE_TRAIN:
        hn = HN(hn.WordEmb_dropout, hn.WordRnn_dropout, hn.SentenceRnn_dropout,
           embedding_regularizer_coefficient=config.embedding_regularizer_coefficient,
           kernel_regularizer_coefficient=config.kernel_regularizer_coefficient,
           rnn=config.rnn)
        save_path = 'char.h5'
        print('========== char ==========')
        char_models = hn.make_char_models()
        best_validation_accuracy = hn.load(char_models, save_path)
        hn.run_epochs(char_models, save_path, partial_train=True, best_validation_accuracy=best_validation_accuracy)
        os.rename(save_path, best_char_path)
        # transform from saved char model to concat model
        hn.save_best_validation(0)
        hn.make_concat_models()
        hn.concat_models[0].save_weights(concat_save_path)
        clear()
    if config.COMBINED:
        if os.path.exists(best_char_path) and not os.path.exists(concat_save_path):
            print('char model save file is converted to concat model')
            hn = HN(hn.WordEmb_dropout, hn.WordRnn_dropout, hn.SentenceRnn_dropout,
                   embedding_regularizer_coefficient=config.embedding_regularizer_coefficient,
                   kernel_regularizer_coefficient=config.kernel_regularizer_coefficient,
                   rnn=config.rnn)
            char_models = hn.make_char_models()
            hn.load(char_models, best_char_path)
            save_best_validation(0)
            hn.make_concat_models()
            hn.concat_models[0].save_weights(concat_save_path)
            clear()
        hn = HN(hn.WordEmb_dropout, hn.WordRnn_dropout, hn.SentenceRnn_dropout,
           embedding_regularizer_coefficient=config.embedding_regularizer_coefficient,
           kernel_regularizer_coefficient=config.kernel_regularizer_coefficient,
           rnn=config.rnn)
        print('========== concat ==========')
        concat_models = hn.make_models()
        if not config.SKIP_PRE_TRAIN:
            concat_models = hn.make_concat_models()
            best_validation_accuracy = hn.load(concat_models, concat_save_path)
        else:
            print('pre-train skipped')
        hn.run_epochs(concat_models, concat_save_path, best_validation_accuracy=best_validation_accuracy)

