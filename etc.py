
# coding: utf-8

# In[1]:

import os, numpy, csv, sys
import numpy as np
from random import shuffle, randrange
from myLib import Timer
from nltk import word_tokenize, sent_tokenize
from pprint import pprint
from datetime import datetime
from path import Path

class Data_split:
    def __init__(self, train=False):
        self.data = []
        self.train = train
    class Document:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.tokenized = []
            self.bucket_i = None                            
    def __call__(self):
        return self.data
    def __getitem__(self, i):
        return self.data[i]
    def __len__(self):
        return len(self.data)
    def make_small_data(self, filename):
        if os.path.exists(filename):
            print(filename, 'already exists.')
            return
        with open(filename, 'w') as f:
            for document in self.data:
                if document.n_sentences < 5 and document.max_sent_len < 30:
                    f.write('x' + '\t\t' + 'x' + '\t\t' + str(document.y+1) + '\t\t' + document.x + '\n')
def parse_ss(file_path, data_split):
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            x, y = line[6], int(line[4])-1
            data_split.data += [data_split.Document(x, y)]
def get_2D_list(document, sentence_separator):
    if document.tokenized:
        return document.tokenized
    else:
        return [sentence.strip(' ').split(' ') for sentence in document.x.split(sentence_separator)]

def write_to_ss_file(file_path, data_split):
    if type(data_split) is Data_split:
        data_split = data_split()
    assert type(data_split) is list
    assert path_extension(file_path) == '.ss'
    with open(file_path, 'w') as ss_f:
        for sample in data_split:
            ss_f.write('x' + '\t\t' + 'x' + '\t\t' + str(sample.y + 1) + '\t\t')
            ss_f.write(sample.x + '\n')
def parse(file_path):
    reviews=[]
    labels=[]
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            labels.append(int(line[4]) - 1)
            reviews.append(line[6])
    return labels, reviews

def text(sentences):
    string=''
    for sentence in sentences:
        for word in sentence:
            string+=word + ' '
    return string
def path_extension(path):
    return os.path.splitext(path)[1]

def get_mini(data, batch_size, shuffle=True):
    order = numpy.arange(len(data), dtype=numpy.int32)
    if shuffle:
        numpy.random.shuffle(order)
    for j in range(0, len(order), batch_size):
        batch = []
        for k in range(j, min(len(order), j + batch_size)):
            batch += [data[order[k]]]
        yield batch

class Dataset:
    '''
    self.dataset.bucketize2(BUCKET_DIVISION, MEMORY)
    self.dataset.import_word_embedding(EMBEDDING_DIM=200)
    if char
        make_character_embedding_index()

    '''
    def __call__(self):
        return self.data
    def __getitem__(self, i):
        return self.data[i]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    
    def __init__(self, log, path,
                 word_coverage, save_dir=None, non_test_flag=True,
                 MAX_WORD_LENGTH=None, MAX_N_CHARACTER=None,
                 max_N_of_sentences=None, max_sent_length=None,
                 clip_batch_size=False, as_text=True, set_min_batch_size=None, bucket_coverage=None):
        if type(path) is str and os.path.isdir(path):
            path = Path(directory_path=path, name=None, glove=True, year=None)
        self.save_dir, self.log = save_dir, open(log, 'a', 1)
        self.max_batch_size, self.set_min_batch_size, self.bucket_coverage             = 999999, set_min_batch_size, bucket_coverage
        self.global_batch_size = None
        self.word_coverage = word_coverage
        self.MAX_N_CHARACTER = MAX_N_CHARACTER
        self.MAX_WORD_LENGTH = MAX_WORD_LENGTH
        self.as_text = as_text
        self.path = path
        self.non_test_flag = non_test_flag
        self.clip_batch_size = clip_batch_size
        self.sentence_separator = '<sssss>'
        self.word_separator = ' '
        self.mem_test = False
        self.class_set = set()
        self.train = Data_split(train=True)
        self.validation = Data_split()
        self.test = Data_split()
        self.debug = 0
        _, extension = os.path.splitext(self.path.train)
        if extension == '.ss':
            parse_ss(self.path.train, self.train)
            parse_ss(self.path.test, self.test)
            if self.path.validation:
                parse_ss(self.path.validation, self.validation)
            else:
                self.split_validation(self.train, self.validation, self.path.train)
            
        elif extension == '.csv':
            self.parse_init_csv(self.path.train, self.train)
            self.parse_init_csv(self.path.test, self.test)
            if self.path.validation:
                self.parse_init_csv(self.path.validation, self.validation)
            else:
                self.split_validation(self.train, self.validation, self.path.train)
                
        def count_each_sample(data_split):
            for document in data_split():
                sentences = get_2D_list(document, self.sentence_separator)
                document.length, document.max_sent_len = 0, 0
                document.n_sentences = len(sentences)
                for sentence in sentences:
                    document.max_sent_len = max(document.max_sent_len, len(sentence))
        count_each_sample(self.train)
        count_each_sample(self.validation)
        count_each_sample(self.test)
        
        self.remove_the_exceeded(max_N_of_sentences, max_sent_length)
        
        print('len(self.train()) = {}   len(self.validation()) = {}   len(self.test()) = {}'.format(
                len(self.train()), len(self.validation()), len(self.test())))
        self.data = self.train() + self.validation() + self.test()
        self.non_test = self.train() + self.validation()
        self.data_initial_process()
        if not (0 in self.class_set):
            for document in self.data:
                document.y -= 1
        self.n_classes = len(self.class_set)
        self.max_sentence_number = 0
        self.max_sentence_length = 0
        self.MAX_LENGTH = 0   # only words

        def count_maxima(self, data_split):
            for document in data_split():
                sentences = get_2D_list(document, self.sentence_separator)
                self.max_sentence_number = max(self.max_sentence_number, len(sentences))
                document.length = 0
                for sentence in sentences:
                    self.max_sentence_length = max(len(sentence), self.max_sentence_length)
                    document.length += len(sentence)
                self.MAX_LENGTH = max(self.MAX_LENGTH, document.length)
        count_maxima(self, self.train)
        count_maxima(self, self.validation)
        count_maxima(self, self.test)
        self.char_embedding_index = None
        self.batch_size = None
        self.word_n = []
        
    def make_small_data(self):
        self.train.make_small_data('train.ss')
        self.validation.make_small_data('validation.ss')
        self.test.make_small_data('test.ss')
        
    def split_validation(self, train, validation, train_path):
        directory_path = os.path.dirname(train_path)
        root, extension = os.path.splitext(train_path)
        if extension == '.csv':
            train_path = root + '.ss'
            assert os.path.exists(train_path)
        else:
            assert extension == '.ss'
        validation_path = os.path.join(directory_path, 'validation.ss')
        if os.path.exists(validation_path):
            print('load:', validation_path)
            parse_ss(validation_path, self.validation)
        else:
            for _ in range(len(self.test())):
                i = randrange(len(train()))
                validation.data += [train.data[i]]
                train.data.remove(train.data[i])
            write_to_ss_file(validation_path, validation)
            write_to_ss_file(train_path, train)
    def parse_init_csv(self, file_path, data_split):
        root, extension = os.path.splitext(file_path)
        ss_path = root + '.ss'
        if os.path.exists(ss_path):
            print('load', ss_path)
            parse_ss(ss_path, data_split)
        else:
            with open(file_path, 'r') as f:
                for line in csv.reader(f, delimiter=','):
                    y = int(line[0])-1
                    tokenized, x = [], ''
                    for line_text in line[1:]:
                        for sentence in sent_tokenize(line_text):
                            tokenized += [word_tokenize(sentence)]
                            for word in tokenized[-1]:
                                if word.find(' ') != -1:
                                    print('space inside the word:  ', data_split.data[-1].tokenized[-1])
                                x += (word + ' ')
                            x += (self.sentence_separator + ' ')
                    data_split.data += [data_split.Document(x, y)]
                    if not self.as_text:
                        data_split.data[-1].tokenized = tokenized
            write_to_ss_file(ss_path, data_split)
    def remove_the_exceeded(self, max_N_of_sentences, max_sent_length):
        if max_N_of_sentences is not None and max_sent_length is not None:
            def remove_the_exceeded_in_split(split):
                new_data_list = []
                for sample in split():
                    if sample.n_sentences <= max_N_of_sentences and sample.max_sent_len <= max_sent_length:
                        new_data_list += [sample]
                split.data = new_data_list
            remove_the_exceeded_in_split(self.train)
            remove_the_exceeded_in_split(self.validation)
            remove_the_exceeded_in_split(self.test)
                
    def data_initial_process(self):
        for document in self.data:
            self.class_set.add(document.y)
            document.sentence_lengths = []
            for sentence in get_2D_list(document, self.sentence_separator):
                document.sentence_lengths += [len(sentence)]
    def get_non_test_data(self):
        return self.non_test if self.non_test_flag else self.data
                
    def calc_word_count(self):
        print('start word count ...')
        self.n_word, self.word_count = 0, {}
        for document in self.get_non_test_data():
            for sentence in get_2D_list(document, self.sentence_separator):
                for word in sentence:
                    self.n_word += 1
                    if self.word_count.get(word):
                        self.word_count[word] += 1
                    else:
                        self.word_count[word] = 1
        print('... end word count.')
        self.sorted_words = [x[0] for x in sorted(self.word_count.items(), key=lambda x:x[1], reverse=True)]
        # print('end sort')
        # convert list to dict
        # self.sorted_word_count_dict = {x[0]:x[1] for x in self.sorted_word_count}
                    
    def make_character_embedding_index(self):
        print('make_character_embedding_index(')
        self.ch_dict = {}
        self.max_word_length = 0
        for document in self.get_non_test_data():
            for sentence in get_2D_list(document, self.sentence_separator):
                for word in sentence:
                    self.max_word_length = max(self.max_word_length, len(word))
                    for ch in word:
                        if self.ch_dict.get(ch):
                            self.ch_dict[ch] += 1
                        else:
                            self.ch_dict[ch] = 1
        sorted_char_list = sorted(self.ch_dict.items(), key=lambda x:x[1])
        self.char_embedding_index = {ch:i+2 for i, ch in enumerate(self.ch_dict)}
        sorted_character_count = sorted(self.ch_dict.items(), key=lambda x:x[1], reverse=True)[:self.MAX_N_CHARACTER]
        self.char_embedding_index = {ch:i+2 for i, (ch, count) in enumerate(sorted_character_count)}
        self.char_embedding_index['no'] = 0
        self.char_embedding_index['truncated'] = 1      # also can be used as <unknown>
        if True: # random uniform initialize
            EMBEDDING_DIM = self.MAX_N_CHARACTER
            positive_width = 0.5 / EMBEDDING_DIM
            self.char_embedding_matrix = [[0]*EMBEDDING_DIM,
                      np.random.uniform(-positive_width, positive_width, (EMBEDDING_DIM))]
            for i in range(2, len(self.char_embedding_index)):
                self.char_embedding_matrix += [np.random.uniform(-positive_width, positive_width, (EMBEDDING_DIM))]
            self.char_embedding_matrix = np.array(self.char_embedding_matrix)
        else:  # one-hot initialize
            self.char_embedding_matrix = np.zeros((len(self.char_embedding_index), 
                      len(self.char_embedding_index)-2), dtype=np.float32)
            for i in range(2, len(self.char_embedding_index)):
                self.char_embedding_matrix[i][i-2] = 1
        print('self.char_embedding_index', self.char_embedding_index)
        print('self.char_embedding_matrix', self.char_embedding_matrix)
        
    def bucketize(self, interval, memory):
        def for_each_datasplit(self, data_split):
            self.bucket_bounds = []
            self.bucket_batch_size = []
            data_split.buckets = []
            # make buckets and bounds by input 'interval'
            for bucket_bound in range(interval, self.MAX_LENGTH, interval):
                self.bucket_bounds += [bucket_bound]
                self.bucket_batch_size += [memory//bucket_bound]
                data_split.buckets += [[]]
            # insert to bucket
            for document in data_split():
                for i, bucket_bound in enumerate(self.bucket_bounds):
                    if document.length <= bucket_bound:
                        data_split.buckets[i] += [document]
                        break
        for_each_datasplit(self, self.train)
        for_each_datasplit(self, self.validation)
        for_each_datasplit(self, self.test)

    def bucketize2(self, n, memory, fixed_batch_size=None):
        self.log.write('bucketize2(\n')
        sentence_number_interval = self.max_sentence_number//n + 1
        sentence_length_interval = self.max_sentence_length//n + 1
        self.log.write('max: {}, {}\n'.format(self.max_sentence_number, self.max_sentence_length))
        self.log.write('sentence_number_interval, sentence_length_interval: {}, {}\n'.format(sentence_number_interval, sentence_length_interval))
        self.bucket_bounds = []
        self.bucket_batch_size = []
        sum_debug = 0
        total_bucket_info = []
        for sentence_number_bound in range(0, self.max_sentence_number+1, sentence_number_interval):
            for sentence_length_bound in range(0, self.max_sentence_length+1, sentence_length_interval):
                bucket_bound = (sentence_number_bound + sentence_number_interval, 
                                 sentence_length_bound + sentence_length_interval)
                documents = []
                for document in self.data:
                    if (document.n_sentences < sentence_number_bound + sentence_number_interval
                            and document.n_sentences >= sentence_number_bound 
                            and document.max_sent_len < sentence_length_bound + sentence_length_interval
                            and document.max_sent_len >= sentence_length_bound):
                        documents += [document]
                sum_debug += len(documents)
                if len(documents) >= 1:
                    bucket_size = 1   # calculate sample size
                    for bucket_length in bucket_bound:
                        bucket_size *= bucket_length
                    batch_size = memory//bucket_size
                    total_bucket_info += [[bucket_bound, batch_size, len(documents)]]
                    if False and self.min_batch_size and self.min_batch_size > 0: # deprecated: calculate global_batch_size by threshold
                        if batch_size >= self.set_min_batch_size:
                            self.global_batch_size = min(batch_size, self.global_batch_size)
                    if fixed_batch_size:
                        self.bucket_batch_size += [fixed_batch_size]
                    elif batch_size == 0:
                        print('no memory exception   memory, bucket_bound =', memory, bucket_bound)
                        sys.exit(1)
                    else:
                        self.bucket_batch_size += [batch_size]
                    # add to buckets
                    self.bucket_bounds += [bucket_bound]
                    for document in documents:
                        document.bucket_i = len(self.bucket_bounds) - 1
        # determine set_min_batch_size
        # sort by bucket's batch_size
        total_bucket_info = sorted(total_bucket_info, key=lambda x:x[1], reverse=True)
        # print bucket_info
        if self.save_dir is not None:
            with open(os.path.join(self.save_dir, 'bucket_info.txt'), 'w') as f:
                n = 0
                for bucket_info in total_bucket_info:
                    n += bucket_info[2]  # cumulative sum of the number of samples
                    f.write('{}  {:.5f}\n'.format(bucket_info, n/len(self.data)))
        # truncate bucket ?
        n_bucket, n = 0, 0
        for bucket_info in total_bucket_info:
            batch_size = bucket_info[1]
            if self.global_batch_size and self.global_batch_size > batch_size:
                break
            n += bucket_info[2]  # cumulative sum of the number of samples
            n_bucket += 1
            bucket_coverage = n/len(self.data)
            if self.set_min_batch_size and bucket_coverage >= self.bucket_coverage:
                self.global_batch_size = batch_size
        self.log.write('len(self.data) = {}   sum_debug = {}\n'.format(len(self.data), sum_debug))
        self.log.write('#(buckets) = {}\n'.format(n_bucket))
        def for_each_datasplit(self, data_split):
            data_split.buckets = []
            for _ in self.bucket_bounds:
                data_split.buckets += [[]]
            for document in data_split():
                data_split.buckets[document.bucket_i] += [document]
        for_each_datasplit(self, self.train)
        for_each_datasplit(self, self.validation)
        for_each_datasplit(self, self.test)

#     def get_batch(self, data_split, CHAR=False):
#         bucket_order = numpy.arange(len(data_split.buckets), dtype=numpy.int32)
#         numpy.random.shuffle(bucket_order)
#         for i in bucket_order:
#             documents = data_split.buckets[i]
#             bucket_batch_size = self.bucket_batch_size[i]
#             bucket_bound = self.bucket_bounds[i]
#             if len(documents) >= 1:
#                 print("len(documents) =", len(documents), "  bucket_bound =", bucket_bound, 
#                       '   bucket_batch_size: ', bucket_batch_size)
#                 for batch in self.get_batch_from_bucket(documents, bucket_batch_size, bucket_bound, CHAR):
#                     yield len(documents), i, batch
                    
    def get_bucket(self, data_split, shuffle=True):
        bucket_order = numpy.arange(len(data_split.buckets), dtype=numpy.int32)
        if shuffle:
            numpy.random.shuffle(bucket_order)
        for i in bucket_order:
            documents = data_split.buckets[i]
            bucket_batch_size = self.bucket_batch_size[i]
            bucket_bound = self.bucket_bounds[i]
            if len(documents) >= 1:
                if self.set_min_batch_size and data_split.train:
                    if bucket_batch_size < self.global_batch_size:
                        continue  # skip small bucket when global_batch_size is defined
                    else:
                        bucket_batch_size = self.global_batch_size
                batch_size = min(self.max_batch_size, bucket_batch_size) if self.clip_batch_size                             else bucket_batch_size
                self.log.write("{}  len(documents) = {}, bucket_bound = {}, batch_size: {}\n".format(
                    datetime.now(), len(documents), bucket_bound, batch_size))
                for batch in self.get_batch_from_bucket(documents, batch_size, shuffle):
                    if self.set_min_batch_size and data_split.train and len(batch) != batch_size:
                        continue  # skip insufficient batch if set_min_batch_size
                    yield len(documents), i, batch

    def get_batch_from_bucket(self, documents, batch_size, shuffle):
        for mini in get_mini(documents, batch_size, shuffle):
            yield mini
            if self.mem_test:
                break
                
    def get_1D_array(self, data_split, batch_size, max_len):
        for mini in get_mini(data_split, batch_size):
            x = numpy.zeros((len(mini), max_len), numpy.int)
            y = numpy.zeros(len(mini), numpy.int)
            for i, document in enumerate(mini):
                j = 0
                y[i] = document.y
                for sentence in get_2D_list(document, self.sentence_separator):
                    for word in sentence:
                        x[i, j] = self.get_word_emb_i(word)
                        j += 1
                        if j >= max_len:
                            break
                    if j >= max_len:
                        break
            yield x, y
            
    def get_array(self, data_split, CHAR=False):
        for n_batch, i, batch in self.get_bucket(data_split):
            bucket_batch_size = self.bucket_batch_size[i]
            bucket_bound = self.bucket_bounds[i]
            ys = numpy.zeros((bucket_batch_size))
            if type(bucket_bound) is int:
                xs = numpy.zeros((bucket_batch_size, bucket_bound), numpy.int)
            else:
                xs = numpy.zeros((bucket_batch_size, *bucket_bound), numpy.int)
            if CHAR:
                exists = numpy.zeros((bucket_batch_size, *bucket_bound), numpy.float32)
                known = numpy.zeros((bucket_batch_size, *bucket_bound), numpy.float32)
                chs = numpy.zeros((bucket_batch_size, *bucket_bound, self.MAX_WORD_LENGTH), numpy.int)
            sentence_lengths_list = []
            for l, document in enumerate(batch):
                sentence_lengths_list += [document.sentence_lengths]
                ys[l] = document.y
                if type(bucket_bound) is int:
                    xs[l,:] = self.text_to_words(bucket_bound, get_2D_list(document, self.sentence_separator))
                else:
                    xs[l,:] = self.text_to_2D_array(bucket_bound, get_2D_list(document, self.sentence_separator))
                    if CHAR:
                        chs[l,:] = self.make_char_array(bucket_bound, get_2D_list(document, self.sentence_separator))
                        exists[l, :], known[l, :] = self.get_exists_and_known(bucket_bound, get_2D_list(document, self.sentence_separator))
            l += 1
            if CHAR:
                xs = [exists, known, xs, chs]
            self.debug += 1
            if False and self.debug<2:
                print('batch[0].x:', batch[0].x)
                print('exists=', exists)
                print('known=', known)
            yield n_batch, i, l, xs, ys, sentence_lengths_list
            
    def get_word_emb_i(self, word):
        index = self.word_embedding_index.get(word)
        return index if index else 1   # 1: <unknown>
    
    def text_to_words(self, bucket_bound, tokenized):
        x = numpy.zeros((bucket_bound), numpy.int)
        word_i = 0
        for sentence in tokenized:
            for word in sentence:
                x[word_i] = self.get_word_emb_i(word)
                word_i += 1
        return x
    
    def text_to_2D_array(self, bucket_bound, tokenized):
        x = numpy.zeros(bucket_bound, numpy.int)
        for i, sentence in enumerate(tokenized):
            for j, word in enumerate(sentence):
                x[i, j] = self.get_word_emb_i(word)
        return x
    
    def make_char_array(self, bucket_bound, tokenized):
        x = numpy.zeros((*bucket_bound, self.MAX_WORD_LENGTH), numpy.int)
        for i, sentence in enumerate(tokenized):
            for j, word in enumerate(sentence):
                for k, ch in enumerate(word):
                    if k == self.MAX_WORD_LENGTH - 1:
                        break
                    index = self.char_embedding_index.get(ch)
                    x[i, j, k] = index if index else 1
                x[i,j,k] = 1 # mark the end of the word
        return x
    
    def get_exists_and_known(self, bucket_bound, tokenized):
        exists = numpy.zeros(bucket_bound, numpy.float32)
        known = numpy.zeros(bucket_bound, numpy.float32)
        for i, sentence in enumerate(tokenized):
            for j, word in enumerate(sentence):
                known[i, j] = 1 if word in self.word_embedding_index else 0
                exists[i, j] = 1
        return exists, known
    
    def get_small_bucket(self, data_split):
        documents = data_split.buckets[0]
        bucket_batch_size = self.bucket_batch_size[0]
        bucket_bound = self.bucket_bounds[0]
        if documents >= 1:
            for batch in self.get_batch_from_bucket(documents, bucket_batch_size, bucket_bound):
                yield len(documents), 0, batch
                
    def get_last_bucket(self, data_split):
        documents = data_split.buckets[-1]
        bucket_batch_size = self.bucket_batch_size[-1]
        bucket_bound = self.bucket_bounds[-1]
        if documents >= 1:
            for batch in self.get_batch_from_bucket(documents, bucket_batch_size, bucket_bound):
                yield len(documents), len(data_split.buckets)-1, batch

    def test_bucketize(self, interval, memory):
        def for_each_datasplit(self, data_split):
            self.bucket_bounds = []
            self.bucket_batch_size = []
            data_split.buckets = []
            # make buckets and bounds by input 'interval'
            for bucket_bound in range(interval, self.MAX_LENGTH, interval):
                bucket_bound = interval
                self.bucket_bounds += [bucket_bound]
                self.bucket_batch_size += [memory//bucket_bound]
                data_split.buckets += [[]]
            # insert to bucket
            print(self.bucket_bounds)
            for document in data_split():
                for i, bucket_bound in enumerate(self.bucket_bounds):
                    if document.length <= bucket_bound:
                        data_split.buckets[i] += [document]
                        break
            for i in range(len(data_split.buckets)):
                if i>0:
                    data_split.buckets[i] = data_split.buckets[0]
            for bucket in data_split.buckets:
                print(len(bucket))
        for_each_datasplit(self, self.train)
        for_each_datasplit(self, self.validation)
        for_each_datasplit(self, self.test)
        

    def get_entire_array(self):
        indices, data = self.indices, self.data
        xs, _, ys, _, _, _ = self.make_array(0, len(data))
        return xs, ys

#     def get_batch(self):
#         indices, data = self.indices, self.data
#         numpy.random.shuffle(indices)
#         print(indices)
#         for start in range(0, len(self.data), self.batch_size):
#             end = start + self.batch_size if start + self.batch_size <= len(data)  else len(data)
#             yield self.make_array(start, end)
            
    
    def make_array(self, start, end):
        indices, data = self.indices, self.data
        size = end - start
        xs = numpy.zeros((size, self.max_sentence_number, self.max_sentence_length), numpy.int)
        if self.MAX_WORD_LENGTH>0:
            chs = numpy.zeros((size, self.max_sentence_number, self.max_sentence_length, self.MAX_WORD_LENGTH), numpy.int)
        else:
            chs = None
        ys = numpy.zeros((size))
        sentence_numbers = numpy.ones(size, numpy.int)
        sentence_lengths = numpy.ones((size, self.max_sentence_number), numpy.int)
        for i in range(start, end):
            batch_i = i - start
            document = data[indices[i]]
            ys[batch_i] = document.y
            text = document.x
            sentences = text.split('<sssss>')
            sentence_numbers[batch_i] = len(sentences) if len(sentences)<=self.max_sentence_number else  self.max_sentence_number
            for j, sentence in enumerate(sentences):
                if j>= self.max_sentence_number:
                    break
                sentence = sentence.strip(' ').split(' ')
                sentence_lengths[batch_i][j] = len(sentence) if len(sentence)<=self.max_sentence_length else self.max_sentence_length
                for k, word in enumerate(sentence):
                    if k>= self.max_sentence_length:
                        break
                    xs[batch_i,j, k] = self.get_word_emb_i(word)
                    
                    for l, ch in enumerate(word):
                        if l>= self.MAX_WORD_LENGTH:
                            break
                        index = self.char_embedding_index.get(ch)
                        if index:
                            chs[batch_i, j, k, l] = index
                        else:
                            chs[batch_i,j, k] = 1 # <unk>                        
        return xs, chs, ys, sentence_numbers, sentence_lengths, size
                        
    def get_batch_from_array(self):
        if not self.xs:
            self.xs, self.ys, self.sentence_numbers, self.sentence_lengths, _ = self.make_array(0, len(self.data))
        indices, data = self.indices, self.data
        numpy.random.shuffle(indices)
        for start in range(0, len(self.data), batch_size):
            end = start + self.batch_size if start + self.batch_size <= len(data)  else len(data)
            yield self.xs[indices[start:end]], self.ys[indices[start:end]],                     self.sentence_numbers[indices[start:end]], self.sentence_lengths[indices[start:end]],                     end-start

    def import_word_embedding(self, EMBEDDING_DIM):
        self.calc_word_count()
        n_line = 0
        with open(self.path.word_embedding, 'r') as f:
            for line in f:
                n_line += 1
        embedding_matrix_size = n_line + 2
        positive_width = 0.5 / EMBEDDING_DIM
        self.word_embedding_index = {}
        embeddings = [[0]*EMBEDDING_DIM, np.random.uniform(-positive_width, positive_width, (EMBEDDING_DIM))]
        n = 2
        n_covered_word = 0
        word_emb_dict = {}
        with open(self.path.word_embedding, 'r') as f:
            for line in f:
                line = line.split(' ')
                assert len(line[1:]) == EMBEDDING_DIM
                word_emb_dict[line[0]] = line[1:]
        with open('word-list.txt', 'w') as F:
            for word in self.sorted_words:
                F.write('{}\n'.format(word))
        for word in self.sorted_words:
            vec = word_emb_dict.get(word)
            if vec is not None:
                n_covered_word += self.word_count[word]
                embeddings += [vec]
                self.word_embedding_index[word] = n
                n += 1
                if self.word_coverage and n_covered_word >= (self.word_coverage * self.n_word):
                    break
        print('word number coverage: {} / {} = {}'.format(n_covered_word, self.n_word, n_covered_word/self.n_word))
        self.word_embedding_matrix = np.array(embeddings, dtype=np.float32)
        print('word embedding matrix size = {}'.format(len(self.word_embedding_matrix)))

    def print_statistics(self):
        max_sentence_number, max_len, sum_n_sent, sum_len = 0, 0, 0, 0
        for document in self:
            sentences = get_2D_list(document, self.sentence_separator)
            max_sentence_number = max(max_sentence_number, len(sentences))
            max_len = max(max_len, document.length)            
            sum_n_sent += len(sentences)
            sum_len += document.length
        print('documents = {}   avg_n_sent = {:.1f}   max_n_sent = {}   avg_len = {:.1f}   max_len = {}'.format(
                len(self), sum_n_sent/len(self), max_sentence_number, sum_len/len(self), max_len))

