import os, sys
'''
glove: sometimes glove becomes the path of the glove embedding matrix
'''
class Path:
    def __init__(self, name, glove, directory_path=None, year=None):
        self.directory = directory_path
        self.validation = None
        self.glove = glove
        if name == 'imdb':
            self.emnlp_imdb()
        elif name == 'small':
            self.small_data()
        elif name == 'emnlp':
            if not year:
                print('feed year !!')
                sys.exit(1)
            self.make_EMNLP_yelp(year)
        elif name == 'nips':
            self.make_NIPS_yelp_csv()
        elif name=='yahoo':
            self.yahoo()
        elif name=='amazon':
            self.amazon()
        elif name=='test_csv':
            self.test_csv()
        elif os.path.isdir(directory_path):
            print('data-dir.: {}'.format(self.directory))
            self.train = os.path.join(self.directory, 'train.ss')
            self.validation = os.path.join(self.directory, 'validation.ss')
            self.test = os.path.join(self.directory, 'test.ss')
            assert os.path.isfile(self.train) and os.path.isfile(self.validation) and os.path.isfile(self.test)
            self.word_embedding_path('aaaaaaa')
        else:
            print('invalid path name')
            sys.exit(1)
    def word_embedding_path(self, path):
        if not self.glove:
            self.word_embedding = os.path.join(self.directory, path)
        else:
            if os.path.isfile(self.glove):
                self.word_embedding = os.path.join(self.glove)
            else:
                self.word_embedding = os.path.join(self.directory, 'glove.6B.200d.txt')
    def complete_emnlp_path(self, prefix):
        self.train = prefix + '-train.txt.ss'
        self.validation = prefix + '-dev.txt.ss'
        self.test = prefix + '-test.txt.ss'
    def small_data(self):
        print('use small dataset')
        self.make_EMNLP_yelp(2013)
        self.train = 'data/small/train.ss'
        self.validation = 'data/small/validation.ss'
        self.test = 'data/small/test.ss'
    def emnlp_imdb(self):
        print('emnlp_imdb', 'dataset is used')
        self.word_embedding_path('aaaaaaa')
        path_prefix = os.path.join(self.directory, 'imdb')
        self.complete_emnlp_path(path_prefix)
    def make_EMNLP_yelp(self, year):
        print('make_EMNLP_yelp', year, 'dataset is used')
        year = str(year)
        self.word_embedding_path('yelp_' + year + '.skip_grams.6.txt')
        path_prefix = os.path.join(self.directory, 'yelp-' + year)
        self.complete_emnlp_path(path_prefix)
    def make_NIPS_yelp_csv(self):
        print('make_NIPS_yelp_csv', 'dataset is used')
        directory = os.path.join(self.directory, 'NIPS/yelp')
        self.word_embedding_path(os.path.join(directory, 'f.skip-gram.6.txt'))
        self.train = os.path.join(directory, 'train.csv')
        self.test = os.path.join(directory, 'test.csv')
    def yahoo(self):
        print('yahoo dataset')
        directory = os.path.join(self.directory, 'yahoo')
        self.word_embedding_path('aaaaa')
        self.train = os.path.join(directory, 'train.ss')
        self.validation = os.path.join(directory, 'validation.ss')
        self.test = os.path.join(directory, 'test.ss')
    def amazon(self):
        print('amazon dataset')
        directory = os.path.join(self.directory, 'amazon')
        self.word_embedding_path('aaaaa')
        self.train = os.path.join(directory, 'train.ss')
        self.validation = os.path.join(directory, 'validation.ss')
        self.test = os.path.join(directory, 'test.ss')
    def test_csv(self):
        print('test_csv dataset')
        directory = os.path.join(self.directory, 'test_csv')
        self.word_embedding_path('aaaaa')
        self.train = os.path.join(directory, 'train.csv')
        self.test = os.path.join(directory, 'test.csv') 
