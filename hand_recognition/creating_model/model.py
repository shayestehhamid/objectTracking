import numpy
from sklearn.lda import LDA

data = numpy.loadtxt("data.data", dtype=numpy.uint8)
numpy.random.shuffle(data)
print data.shape
cls = data[:, 0]
data = data[:, 1:]
print "sum", numpy.sum(cls)
cls_test = cls[0:100]
data_test = data[0:100, :]

cls_train = cls[100:]
data_train = data[100:, :]
print data_train.shape, cls_train.shape
clf = LDA()
clf.fit(data_train, cls_train)
print "fitted"
correct = 0
all = 0
ans = 0
for row in xrange(data_test.shape[0]):
    all += 1
    pred = clf.predict(data_test[row, :])
    ans += cls_test[row]
    correct += 1 if pred == cls_test[row] else 0
print "ans", ans
print all, correct, float(correct)/all

