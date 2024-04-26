import string
import os
import random

DEFAULT_DATA_DIR = 'data'

class ToyFirstCharModel():
    def __init__(self, data_dir = DEFAULT_DATA_DIR):
        self.data_dir = os.path.join(os.getcwd(), data_dir)

    def get_random_string_with_answer(self, length, valid_chars):
        sequence = ''.join(random.choices(valid_chars, k= length))
        return sequence + " " + sequence[0] + "\n"

    def generate_data(self, base_size = 100000, base_length = 100, \
                      valid_chars = string.ascii_uppercase + string.digits, datapath = None):
        '''
        Generates data for first character test in the format:

        "sequence of characters" "c"

        where "c", being the answer, is the first character of the sequence

        @param base_size: size of training set
        @param base_length: length of sequence for training set
        @param valid_chars: possible characters included in the sequence
        @param datapath: path to save to
        '''

        if (datapath == None):
            if not os.path.isdir(self.data_dir):
                os.mkdir(self.data_dir)
            datapath = self.data_dir

        charpath = os.path.join(self.data_dir, 'valid_chars.txt')
        with open(charpath, 'w') as f:
            f.write(valid_chars)

        trainpath = os.path.join(self.data_dir, 'first_char_train.txt')
        with open(trainpath, 'w') as f:
            for _ in range(base_size):
                f.write(self.get_random_string_with_answer(base_length, valid_chars))

        devpath = os.path.join(self.data_dir, 'first_char_dev.txt')
        with open(devpath, 'w') as f:
            for _ in range(int(base_size * 0.3)):
                f.write(self.get_random_string_with_answer(base_length, valid_chars))

        long_devpath = os.path.join(self.data_dir, 'first_char_long_dev.txt')
        with open(long_devpath, 'w') as f:
            for _ in range(int(base_size * 0.3)):
                f.write(self.get_random_string_with_answer(base_length * 2, valid_chars))

        longer_devpath = os.path.join(self.data_dir, 'first_char_longer_dev.txt')
        with open(longer_devpath, 'w') as f:
            for _ in range(int(base_size * 0.3)):
                f.write(self.get_random_string_with_answer(base_length * 3, valid_chars))

        testpath = os.path.join(self.data_dir, 'first_char_test.txt')
        with open(testpath, 'w') as f:
            for _ in range(int(base_size * 0.1)):
                f.write(self.get_random_string_with_answer(base_length, valid_chars))
                f.write(self.get_random_string_with_answer(base_length * 2, valid_chars))
                f.write(self.get_random_string_with_answer(base_length * 3, valid_chars))

    def construct_input(self, datapath, char2id, max = None):
        queries = []
        answers = []
        with open(datapath, 'r') as file:
            array = file[:max] if max else file
            for line in array:
                split = line.split()
                ques = [char2id[c] for c in split[0]]
                ans = char2id[split[1]]
                queries.append(ques)
                answers.append(ans)

        return queries, answers

    def load_data(self, reduced = False, datapath = None):
        if datapath == None:
            datapath = self.data_dir

        trainpath = os.path.join(datapath, 'first_char_train.txt')
        devpath = os.path.join(datapath, 'first_char_dev.txt')
        long_devpath = os.path.join(datapath, 'first_char_long_dev.txt')
        longer_devpath = os.path.join(datapath, 'first_char_longer_dev.txt')
        testpath = os.path.join(datapath, 'first_char_test.txt')
        assert(os.path.isfile(trainpath))
        assert(os.path.isfile(devpath))
        assert(os.path.isfile(long_devpath))
        assert(os.path.isfile(longer_devpath))
        assert(os.path.isfile(testpath))

        charpath = os.path.join(self.data_dir, 'valid_chars.txt')
        f = open(charpath, 'r')
        valid_chars = f.readline()
        f.close()
        char2id = {v: i for i, v in enumerate(valid_chars)}

        train_set = self.construct_input(trainpath, char2id, 1000) if reduced else self.construct_input(trainpath, char2id)
        dev_set = self.construct_input(devpath, char2id, 300) if reduced else self.construct_input(devpath, char2id)
        long_dev_set = self.construct_input(long_devpath, char2id, 300) if reduced else self.construct_input(long_devpath, char2id)
        longer_dev_set = self.construct_input(longer_devpath, char2id, 300) if reduced else self.construct_input(longer_devpath, char2id)
        test_set = self.construct_input(testpath, char2id, 300) if reduced else self.construct_input(testpath, char2id)

        return train_set, dev_set, long_dev_set, longer_dev_set, test_set
