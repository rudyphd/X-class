import pandas as pd
import copy
#import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
from astropy.table import Table, vstack
from sklearn.ensemble import RandomForestClassifier

def parse_region_file(reg_file): 
    f = open(reg_file,'r')
    x,y,r =[],[],[]
    for l in f.readlines(): 
        if l[0] not in ['#','\n',' ']:
            xt,yt,rt = l.split('(')[1].split(')')[0].split(',')
            x.append(float(xt));y.append(float(yt));r.append(float(rt))
    return np.asarray(x),np.asarray(y),np.asarray(r)

def get_events(evt,x,y,r=6.0):
    """get events from a position and calculate offset"""
    revt = np.sqrt((evt['x']-x)**2+(evt['y']-y)**2)
    tevt = copy.copy(evt[revt<=r])
    tevt['xoff'] = tevt['x']-x
    tevt['yoff'] = tevt['y']-y
    del tevt['x']
    del tevt['y']
    return tevt

def merge_pos(evt,x,y,lab,r=6.0): 
    for i in range(len(x)): 
        if (i == 0): 
            mevt = get_events(evt,x[i],y[i],r)
            levt = np.zeros(len(mevt))
            levt[:] = lab[i]
        else:
            tmpevt = get_events(evt,x[i],y[i],r)
            tmplevt = np.zeros(len(tmpevt))
            tmplevt[:] = lab[i] 
            mevt = vstack([mevt,tmpevt])
            levt = np.hstack([levt,tmplevt])
            
    return mevt,levt

def build_rfc(evt,lab,rfc = None):
    if (rfc is None): 
        rfc = RandomForestClassifier(n_estimators=200,oob_score=True)
    X = copy.copy(evt.to_pandas())
    Y = copy.copy(lab)
    rfc.fit(X.values,Y)
    return rfc,X.values,Y

def do_rfc(evt,rfc): 
    X = copy.copy(evt.to_pandas())
    Y = rfc.predict(X.values)
    print "{0:0.1f} {1:0.1f} ({2})".format(100.*float(len(np.where(Y==0)[0]))/len(Y),
                                           100.*float(len(np.where(Y==1)[0]))/len(Y),len(Y))
    return Y 
     

#def class_pos(evt,pos,r=10.0): 
#    """classify position for source or bkg"""
    
# read all events 
e = Table.read('Data/evt_1229.fits',hdu=1)
# rid ec of bad columns
ec = copy.copy(e)
badcols = ['status','ccd_id','expno','node_id','chipx','chipy','tdetx','tdety','detx','dety','pi','pha']
for bc in badcols: 
    del ec[bc]
ed = ec.to_pandas()

# read region positions:
b1x,b1y,b1r = parse_region_file('Data/b1_1229.reg')
b2x,b2y,b2r = parse_region_file('Data/b2_1229.reg')
s1x,s1y,s1r = parse_region_file('Data/src1_1229.reg')
s2x,s2y,s2r = parse_region_file('Data/src2_1229.reg')

# using b1,s1 as training, b2,s2 as
trnx,trny = np.hstack((s1x,b1x)),np.hstack((s1y,b1y))
trnl = np.hstack((np.ones(len(s1x)),np.zeros(len(b1x)))) 
trne,trnlab = merge_pos(ec,trnx,trny,trnl)

rfc,X,Y = build_rfc(trne,trnlab) # = RandomForestClassifier(n_estimators=200,oob_score=True)

print "OOB Score: {0}".format(rfc.oob_score_)
sorted(zip(trne.colnames,rfc.feature_importances_),key=lambda q: q[1],reverse=True)

# sources
print "Source Tests"
print "BG%  S% (N)"
for xi,yi in zip(s2x,s2y): 
    lp = do_rfc(get_events(ec,xi,yi),rfc)

# bg
print "Background Tests"
print "BG%  S%"
for xi,yi in zip(b2x,b2y):
    lp = do_rfc(get_events(ec,xi,yi),rfc)


