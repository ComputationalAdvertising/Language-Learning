#!/usr/bin/python
# coding=utf-8

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt

# A: path
'''
train = '/data2/zeus/KaggleCtrContest/trained_data/onehot_pseudoctr_discretization_numerical_combined/encoded_onehot_ctr_disc_num_normal_combined_train.data'
test = '/data2/zeus/KaggleCtrContest/trained_data/onehot_pseudoctr_discretization_numerical_combined/encoded_onehot_ctr_disc_num_normal_combined_test.data'
submission = 'submit_060_f5_L110_L240.csv'
'''

train = '/data0/zeus/KaggleCtrContest/raw_and_encoded_data/onehot_pseudoctr_discretization_numerical_normalization_combined/combined_feature_train.data'
test = '/data0/zeus/KaggleCtrContest/raw_and_encoded_data/onehot_pseudoctr_discretization_numerical_normalization_combined/combined_feature_test.data'
submission = 'submit_066_ftrl_raw_encoded_L130_L240_D25.csv'
save_train_path="/data0/zeus/KaggleCtrContest/raw_and_encoded_data/hash_and_encoded_data/train.data"
save_test_path="/data0/zeus/KaggleCtrContest/raw_and_encoded_data/hash_and_encoded_data/test.data"
'''
train = 'raw_encoded_train.10000'
test = 'raw_encoded_test.1000'
submission = 'submit_raw_encode.csv'
'''
# B: model parameter
alpha = 0.1     # learning rate
beta = 1.0      # smoothing parameter for adaptive learning rate
L1 = 3        # L1 regularization, larger value means more regularized
L2 = 4        # L2 regularization, larger value means more regularized

# C: feature/hash trick
D = 2 ** 25            # number of weights to use
max_index = 35472 + 1
interaction = False     # whether to enable poly2 feature interactions

# D: training/validation
epoch = 1           # learn training data for N passes
holdafter = 9       # data after date N (exclusive) are used as validation
holdout = None      # use every N training instance for holdout validation


#################################################
# class, function, generator definations
#################################################

class ftrl_proximal(object):
    
    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2
        
        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: 
        # z: full weights
        # w: lazy weights
        self.n = [0.] * (D + max_index)
        self.z = [0.] * (D + max_index)
        self.w = {}

    def _indices(self, x):
        '''
            A helper generator that yields the indices in x
            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''
        # first field index of the bias term
        yield 0
        # the yield the normal indices
        for index in x:
            yield index
        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)
            
            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0
        for elem in (['0:1.0'] + sorted(x)):
            index_value = elem.split(':')
            i = int(index_value[0])
            v = float(index_value[1])

            sign = -1.0 if z[i] < 0 else 1.0

            if sign * z[i] <= L1:
                w[i] = 0
            else:
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i] * v
        # cache the current w for update stage
        self.w = w
        # bounded sigmoid function, this is the probability estimation
        return 1.0 / (1.0 + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        # parameter
        alpha = self.alpha
        # model
        n = self.n
        z = self.z
        w = self.w
        # update z and n
        for elem in (['0:1.0'] + sorted(x)):
            index_value = elem.split(':')
            i = int(index_value[0])
            v = float(index_value[1])
            # gradient under logloss
            g = (p - y) * v
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g

def logloss(p, y):
    '''
        FUNCTION: Bounded logloss
        Input:
            p: prediction result
            y: real value
        Output:
            logarithmic loss of p given y
    '''
    p = max(min(p, 1.0 - 1e-15), 1e-15)
    return -log(p) if y == 1.0 else -log(1.0 - p)

def data(path, D):
    for t, row in enumerate(DictReader(open(path))):
        # process id
        ID = row['id']
        del row['id']

        # process clicks
        y = 0.0
        if 'click' in row:
            if row['click'] == '1':
                y = 1.0
            del row['click']

        # extract date
        date = int(row['hour'][4:6])
        # turn hour really into hour, it was originally YYMMDDHH
        row['hour'] = row['hour'][6:]

        # build x
        x = []
        for key in row:
            value = row[key]
            # one-hot encode everything with hash trick
            index = abs(hash(key + '_' + value)) % D
            value = 1.0
            x.append('%d:%0.1f' % (index, value))

        yield t, date, ID, x, y

def get_encoded_test_data(path):
    for t, line in enumerate(open(path).readlines()):
        line_list = line.split(' ')
        if len(line_list) == 0:
            continue
        if line_list[0] == '1.0' or line_list[0] == '0.0':
            y = float(line_list[0])
        else:
            y = line_list[0]
        del line_list[0]
        raw_feature_data = line_list[:22]
        raw_feature_data[0] = raw_feature_data[0][6:]
        encoded_data = line_list[23:]
        x = []
        # process raw data
        for i in range(len(raw_feature_data)):
            index = abs(hash(str(i) + '_' + raw_feature_data[i])) % D
            x.append("%d:%d" % (index, 1))
        # process encoded data
        for index_value in encoded_data:
            index = int(index_value.split(':')[0]) + D
            value = index_value.split(':')[1]
            x.append('%d:%s' % (index, value))

        yield t, x, y

####################################################
# start training
####################################################
start = datetime.now()
# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
# start training
for e in xrange(epoch):
    train_file = open(train)
    train_file.seek(0)
    line = train_file.readline()    # only read one raw at each times
    while line:
        line_list = line.split(' ')
        if len(line_list) == 0:
            continue
        if line_list[24] == '1.0' or line_list[24] == '0.0':
            y = float(line_list[24])
        else:
            y = line_list[24]
        
        raw_data = line_list[2:24]
        raw_data[0] = raw_data[0][6:]      # extract hour
        encoded_data = line_list[25:]
        
        x = []
        # process raw data
        for i in range(len(raw_data)):
            index = abs(hash(str(i) + '_' + raw_data[i])) % D
            x.append("%d:%d" % (index, 1))
        # process encoded data
        for index_value in encoded_data:
            index = int(index_value.split(':')[0]) + (D-1)
            value = index_value.split(':')[1]
            x.append('%d:%s' % (index, value))
        #print '%s\t%s' % (str(y), ','.join(x))
        p = learner.predict(x)
        learner.update(x, p, float(y))
        del line, x
        line = train_file.readline()    # read next raw data
process = datetime.now()
print 'train time: ', (process - start)

####################################################
# start testing
###################################################
test_file = open(test)
with open(submission, 'w') as outfile:
    outfile.write('id,click\n')
    for t, x, y in get_encoded_test_data(test):
        p = learner.predict(x)
        outfile.write('%s,%s\n' % (y, str(p)))
        del t, x, y
'''
    while 1:
        lines = test_file.readlines()
        if not lines:
            break
        for line in lines:
            line_list = line.split(' ')
            if len(line_list) == 0:
                continue
            if line_list[0] == '1.0' or line_list[0] == '0.0':
                y = float(line_list[0])
            else:
                y = line_list[0]
            del line_list[0]
            x = []
            for index_value in line_list:
                x.append(index_value)
            p = learner.predict(x)
            outfile.write('%s,%s\n' % (y, str(p)))
            del line
            del x
'''
end = datetime.now()
print 'total run time: ', (end-start)
