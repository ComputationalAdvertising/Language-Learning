#!/usr/bin/python
# coding = utf-8

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt

# A: path
train = '/data2/zeus/KaggleCtrContest/raw_data/train_data.data'               # path to training file
test = '/data2/zeus/KaggleCtrContest/raw_data/test_data.data'                 # path to testing file
submission = 'submit_063_ftrl_raw_L110_L240_D20.csv'

# B: model parameter
alpha = 0.1     # learning rate
beta = 1.0      # smoothing parameter for adaptive learning rate
L1 = 1.0        # L1 regularization, larger value means more regularized
L2 = 4.0        # L2 regularization, larger value means more regularized

# C: feature/hash trick
D = 2 ** 20             # number of weights to use
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
        # z:
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
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
        #for i in self._indices(x):
        for i in ([0] + x):
            sign = -1.0 if z[i] < 0 else 1.0

            if sign * z[i] <= L1:
                w[i] = 0
            else:
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]
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
        # gradient under logloss
        g = p - y
        # update z and n
        #for i in self._indices(x):
        for i in ([0] + x):
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
            x.append(index)

        yield t, date, ID, x, y

# test
'''
with open('aresult', 'w') as outfile:
    for t, date, ID, x, y in data('atest', 100):
        outfile.write(','.join([str(i) for i in x])
        outfile.write('\n')
'''    
####################################################
# start trianing
####################################################
start = datetime.now()
# initialize outselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
# start training
for e in xrange(epoch):
    for t, date, ID, x, y in data(train, D):
        # step1: get prediction from learner
        p = learner.predict(x)
        learner.update(x, p, y)

####################################################
# start training
####################################################
with open(submission, 'w') as outfile:
    outfile.write('id,click\n')
    for t, date, ID, x, y in data(test, D):
        p = learner.predict(x)
        outfile.write('%s,%s\n' % (ID, str(p)))
