import string
import os
import random

DEFAULT_DATA_DIR = os.path.join(os.getcwd(), 'data')

def get_string(length, valid_nums, fn):
    '''
    Generates one line of data
    Front string delimited by ' ', answer delimited by '  '
    '''
    SEQ_DELIMITER = ' '
    ANS_DELIMITER = '  '
    sequence = random.choices(valid_nums, k=length)
    answer = fn(sequence)
    return SEQ_DELIMITER.join([str(n) for n in sequence]) + ANS_DELIMITER + str(answer) + '\n'

def kmax(arr, k=10):
    '''
    Find the k-th largest element

    @params arr: list of ints
    @params k: k-th largest
    
    '''
    assert(len(arr) > 0)
    if k == 1:
        return max(arr)
    copied = arr.copy()
    for _ in range(k-1):
        copied.remove(max(copied))
    return max(copied)

def first(arr):
    return arr[0]

def write_to_file(length, valid_nums, fn, filepath, lines_n):
    with open(filepath, 'w') as f:
        for _ in range(lines_n):
            f.write(get_string(length, valid_nums, fn))

def read_from_file(filepath, max_lines = None):
    queries = []
    answers = []
    with open(filepath, 'r') as file:
        array = file[:max_lines] if max_lines else file
        for line in array:
            split_line = line.split("  ")
            ques = [int(n) for n in split_line[0].split(" ")]
            ans = int(split_line[1])
            queries.append(ques)
            answers.append(ans)

    return queries, answers

def generate_data(base_size = 100000, base_length = 100, char_n = 10, filepath = None, **kwargs):
    '''
    Generates data for first character test in the format:

    "sequence of ints" "answer int"

    @param base_size: size of training set
    @param base_length: length of sequence for training set
    @param char_n: number of types of characters
    @param filepath: path to save to (just folder name)
    @param kwargs: 

    Kwargs:
    - name: name of fn
    - any extra params
    '''

    valid_nums = range(char_n)
    k = 10

    if (filepath == None):
        filepath = DEFAULT_DATA_DIR
    else:
        filepath = os.path.join(os.getcwd(), filepath)

    if not os.path.isdir(filepath):
        os.mkdir(filepath)

    if kwargs['fn'] == 'first':
        fn = first
    else:
        fn = lambda x: kmax(x, k)

    trainpath = os.path.join(filepath, 'train.txt')
    write_to_file(base_length, valid_nums, fn, trainpath, base_size)

    devpath = os.path.join(filepath, 'dev.txt')
    write_to_file(base_length, valid_nums, fn, devpath, int(base_size * 0.3))

    testpath = os.path.join(filepath, 'test.txt')
    write_to_file(base_length, valid_nums, fn, testpath, int(base_size * 0.1))
    print('Data generation complete!')

def load_data(reduced = False, datapath = None):
    if datapath == None:
        datapath = DEFAULT_DATA_DIR

    trainpath = os.path.join(datapath, 'train.txt')
    devpath = os.path.join(datapath, 'dev.txt')
    testpath = os.path.join(datapath, 'test.txt')
    assert(os.path.isfile(trainpath))
    assert(os.path.isfile(devpath))
    assert(os.path.isfile(testpath))

    train_set = read_from_file(trainpath, 1000) if reduced else read_from_file(trainpath)
    dev_set = read_from_file(devpath, 300) if reduced else read_from_file(devpath)
    test_set = read_from_file(testpath, 300) if reduced else read_from_file(testpath)

    return train_set, dev_set, test_set