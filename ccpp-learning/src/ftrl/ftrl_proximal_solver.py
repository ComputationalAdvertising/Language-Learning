#!/usr/bin/env python
# _*_ coding: utf-8 _*_

from math import exp, log, sqrt
from datetime import datetime
import sys

##############################################
# ftrl
##############################################

class ftrl_proximal_learner(object):

	def __init__(self, alpha, beta, l1, l2, D):
		# parameters
		self.alpha = alpha
		self.beta = beta
		self.l1 = l1
		self.l2 = l2

		# model
		# n:
		# z: full weights
		# w: lazy weights
		self.n = [0.] * D
		self.z = [0.] * D
		self.w = {}

	def predict(self, x):
		# parameters
		alpha = self.alpha
		beta = self.beta
		l1 = self.l1
		l2 = self.l2

		# model
		n = self.n
		z = self.z
		w = {}

		# wTx is the inner product of w and x
		wTx = 0
		for elem in (['0:1.0'] + sorted(x)):
			index_value = elem.split(':')
			if len(index_value) != 2:
				print 'len(index_value) != 2:'.join(index_value)
				continue
			index = int(index_value[0])
			value = float(index_value[1])
			
			sign = -1.0 if z[index] < 0 else 1.0

			if sign * z[index] <= l1:
				w[index] = 0
			else:
				w[index] = (l1 * sign - z[index]) / ((beta + sqrt(n[index])) / alpha + l2)

			wTx += w[index] * value
		# cache the current w for update stage
		self.w = w
		# bounded sigmoid function. probability estimation
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
			if len(index_value) != 2:
				print "update !=2"
				continue
			index = int(index_value[0])
			value = float(index_value[1])
			# gradient
			g = (p - y) * value
			sigma = (sqrt(n[index] + g * g) - sqrt(n[index])) / alpha
			z[index] += g - sigma * w[index]
			n[index] += g * g
			
	# predict, label
	def logloss(self, p, y):
		'''
			FUNCTION: Bounded logloss
			Input:
				p: prediction result
				y: real value
			Output:
				logarithmic loss of p given y
		'''
		pn = max(min(p, 1.0 - 1e-15), 1e-15)
		return -log(pn) if y == 1.0 else -log(1.0 - pn)

##############################################
# training model
##############################################
if len(sys.argv) < 3:
	print >> sys.stderr, "Usage: [training] [testing] [log]"
	sys.exit(-1)

training = open(sys.argv[1])
testing = open(sys.argv[2])
log_file = open(sys.argv[3], 'w')

# model parameters
alpha = 0.1		# learning rate
beta = 1.0 		# smooth parameter for adaptive learning rate
l1 = 3
l2 = 4
D = 149	+ 1		# feature dim

ftrl_learner = ftrl_proximal_learner(alpha, beta, l1, l2, D)
# training phase
log_file.write("[INFO] training beginning ...\n")
start = datetime.now()

training.seek(0)
line = training.readline()
while line:
	line_items = line.strip().split(' ')			# libsvm format
	if 0 == len(line_items):
		continue
	y = float(line_items[0])
	x = line_items[1:]

	# predict
	p = ftrl_learner.predict(x)
	# update
	ftrl_learner.update(x, p, y)

	del line, x
	line = training.readline()

end = datetime.now()
log_file.write("[INFO] training end.\n[INFO] training time: %s\n\n" % (end - start))

# testing phase
logloss_value = 0.0
m = 0
log_file.write("[INFO] testing beginning ...\n")
start = datetime.now()
testing.seek(0)
line = testing.readline()
while line:
	line_items = line.strip().split(' ')
	if 0 == len(line_items):
		continue
	y = float(line_items[0])
	x = line_items[1:]
	p = ftrl_learner.predict(x)
	logloss_value += ftrl_learner.logloss(p, y)
	m += 1
	del line_items, x, y
	line= testing.readline()

end = datetime.now()
log_file.write("[INFO] testing end.\n[INFO]testing time: %s\n\n" % (end - start))
log_file.write("[INFO] logloss in testing: %f\n" % (logloss_value / m))
log_file.write("[INFO] done!!!\n") 
