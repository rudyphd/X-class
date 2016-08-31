import pandas as pd
import copy
#import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
from astropy.table	import Table
from sklearn.ensemble import RandomForestClassifier

#sns.set()
e = Table.read('evt_1229.fits',hdu=1)
ec = copy.copy(e)
s = Table.read('sevt_1229.fits',hdu=1)
b = Table.read('bevt_1229.fits',hdu=1)

badcols = ['status','x','y','ccd_id','expno','node_id','chipx','chipy','tdetx','tdety','detx','dety','pi','pha','pha_ro']
for bc in badcols: 
	del s[bc]
	del b[bc]
	del ec[bc]

sd = s.to_pandas()
bd = b.to_pandas()
ed = ec.to_pandas()

s = Table.read('sevt_1229.fits',hdu=1)
b = Table.read('bevt_1229.fits',hdu=1)

#rez = pd.scatter_matrix(sd[0:100])
#rez = pd.scatter_matrix(bd[0:100])

all_training = pd.concat([sd,bd])
all_labels  = np.zeros(len(all_training))
all_labels[0:len(sd)]=1.0

clf = RandomForestClassifier(n_estimators=200,oob_score=True)
X = copy.copy(all_training.values)
Y = copy.copy(all_labels)
clf.fit(X,Y)

print "OOB Score: {0}".format(clf.oob_score_)
sorted(zip(all_training.columns.values,clf.feature_importances_),key=lambda q: q[1],reverse=True)

e_labels = clf.predict(ed)

plt.clf()

f1 = plt.figure(num='trials')
plt.subplot(1,2,1)
plt.hist2d(s['x'],s['y'],range=[[3200,4500],[3200,4500]],bins=400,cmap=cm.coolwarm)
plt.annotate('SRC',xy=(0.1,0.9),xycoords='axes fraction',color='w')
plt.subplot(1,2,2)
plt.hist2d(b['x'],b['y'],range=[[3200,4500],[3200,4500]],bins=400,cmap=cm.coolwarm)
plt.annotate('BKG',xy=(0.1,0.9),xycoords='axes fraction',color='w')

f2 = plt.figure(num='results')
plt.subplot(1,2,1)
plt.hist2d(e['x'][e_labels==1],e['y'][e_labels==1],range=[[3200,4500],[3200,4500]],bins=400,cmap=cm.coolwarm)
plt.annotate('SRC',xy=(0.1,0.9),xycoords='axes fraction',color='w')

plt.subplot(1,2,2)
plt.hist2d(e['x'][e_labels==0],e['y'][e_labels==0],range=[[3200,4500],[3200,4500]],bins=400,cmap=cm.coolwarm)
plt.annotate('BKG',xy=(0.1,0.9),xycoords='axes fraction',color='w')

f3 = plt.figure(num='spec')
plt.subplot(1,1,1)
plt.hist(e['energy'][e_labels==0],range=[0,12000],bins=100,label='BKG')
plt.hist(e['energy'][e_labels==1],range=[0,12000],bins=100,label='SRC',alpha=0.5)
plt.ylabel('counts')
plt.xlabel('energy')
plt.legend()


