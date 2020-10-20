'''
Test script to validate my dbscan produces similiar results as scikit-learn
'''

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from dbscan import dbscan 

# Create three gaussian blobs to use as our clustering data.
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

###############################################################################
# My implementation of DBSCAN
#

# Run my DBSCAN implementation.
print('Running my implementation...')
my_labels = dbscan(X, eps=0.3, MinPts=10)

###############################################################################
# Scikit-learn implementation of DBSCAN
#

print('Runing scikit-learn implementation...')
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
skl_labels = db.labels_

# Scikit learn uses -1 to for NOISE, and starts cluster labeling at 0. I start
# numbering at 1, so increment the skl cluster numbers by 1.
for i in range(0, len(skl_labels)):
    if not skl_labels[i] == -1:
        skl_labels[i] += 1


###############################################################################
# Did we get the same results?

num_disagree = 0

# Go through each label and make sure they match (print the labels if they 
# don't)
for i in range(0, len(skl_labels)):
    if not skl_labels[i] == my_labels[i]:
        print('Scikit learn:', skl_labels[i], 'mine:', my_labels[i])
        num_disagree += 1

if num_disagree == 0:
    print('PASS - All labels match!')
else:
    print('FAIL -', num_disagree, 'labels don\'t match.' )       