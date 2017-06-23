
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)input_file = "D:\用户目录\Desktop\郭磊\keras\imdb.npz"
from __future__ import absolute_import
from six.moves import zip
import numpy as np
import json
import warnings

def _remove_long_seq(maxlen, seq, label):
    """Removes sequences that exceed the maximum length.

    # Arguments
        maxlen: int, maximum length
        seq: list of lists where each sublist is a sequence
        label: list where each element is an integer

    # Returns
        new_seq, new_label: shortened lists for `seq` and `label`.
    """
    new_seq, new_label = [], []
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return new_seq, new_label


# get_file_path 是文件的地址
def load_data(get_file_path, num_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3, **kwargs):
    # Legacy support
    # out of vocabulary 非推荐术语; 非规范用语;
    if 'nb_words' in kwargs:
        warnings.warn('The `nb_words` argument in `load_data` '
                      'has been renamed `num_words`.')
        num_words = kwargs.pop('nb_words')
        # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
    if kwargs:
        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    path = get_file_path#  输入imdb.npz的地址  get_file_path本来是个函数，远程访问的函数，keras功能函数
    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    # seed( ) 用于指定随机数生成时所用算法开始的整数值，
    # 如果使用相同的seed( )值，则每次生成的随即数都相同，
    # 如果不设置这个值，则系统根据时间来自己选择这个值，
    # 此时每次生成的随机数因时间差异而不同。
    np.random.shuffle(x_train) # 按照相同的随机值进行打乱
    np.random.seed(seed)
    np.random.shuffle(labels_train) # 按照相同的随机值进行打乱

    # 打乱测试集
    np.random.seed(seed * 2)
    np.random.shuffle(x_test)
    np.random.seed(seed * 2)
    np.random.shuffle(labels_test)

    xs = np.concatenate([x_train, x_test])# 把训练集和测试集中输出的特征序列sequence连接起来
    labels = np.concatenate([labels_train, labels_test])# 把训练集和测试集中 标签 连接起来

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                             'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
    else:
        xs = [[w for w in x if (skip_top <= w < num_words)] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    # 最后训练的也是array类型
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)


def get_word_index(get_file_path):
    path = get_file_path
    f = open(path)
    data = json.load(f)
    f.close()
    return data
