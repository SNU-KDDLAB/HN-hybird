memory_fraction = 0.65
import os
os.environ['KERAS_BACKEND']='tensorflow'
if os.environ['KERAS_BACKEND'] == 'theano':
    os.environ['THEANO_FLAGS']='gpuarray.preallocate={}'.format(memory_fraction)
    # import theano
    # theano.config.optimizer='fast_compile'
from att2 import HN
hn = HN(
    [__file__, 'my_keras.py', 'etc.py', 'path.py', 'att2.py'], mem_test=True, word_coverage=0.8, conv_unit_size=256,
    save_dir = 'data', 
    data_directory = '/home/kddlab/dyhong/data/amazon/0.15', data_name=None, year=None,
    mode='max', CHAR=False, pretrain=False, set_min_batch_size=False, bucket_coverage=0.77, adjust_learning_rate=True,
    initial_learning_rate = 0.001, learning_rate_decay = 1.5, optimizer_name = 'rmsprop', optimizer_kwargs={}, clip_batch_size=False,
    memory = 50000, memory_fraction=memory_fraction, BUCKET_DIVISION=15,
    WordEmb_dropout = 0, WordRnn_dropout = 0, SentenceRnn_dropout = 0, char_rnn=False,
    rnn = 'gru',
    entire_char_size=40, CHARACTER_RNN_DIMENSION=50,
    embedding_regularizer_coefficient=0, kernel_regularizer_coefficient=0,
    sec_period = 30, REMAINDER=False, MAX_WORD_LENGTH = 16,
    stack=[1,1], glove='/home/kddlab/dyhong/data/glove.6B.200d.txt', non_test=True, TRAIN=True, mse=False
    )
hn.run()
