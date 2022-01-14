from glob import glob
import mahotas as mh
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from scipy.spatial import distance

app = Flask(__name__)

@app.route('/')
images = glob('C:/Users/1234/Desktop/mahotas실습/sneakers/*.jpg')
images[:5]

images[0][500:-len('000.jpg')]

im = mh.imread(images[0])
im = mh.colors.rgb2gray(im, dtype=np.uint8)
im

mh.features.haralick(im)

features = []
labels = []

start = time.time()

for im in images:
    labels.append(im[500:-len('000.jpg')])
    im = mh.imread(im)
    im = mh.colors.rgb2gray(im, dtype = np.uint8)
    features.append(mh.features.haralick(im).ravel())

print(f'fit time : {time.time() - start}')

features = np.array(features)
labels = np.array(labels)
#print(features)

clf = Pipeline([('preproc', StandardScaler()),('classifier', LogisticRegression())])

#cv = LeaveOneOut()

#scores = cross_val_score(clf, features, labels, cv=cv)
#print(scores)

#print('Accuracy: {:2%}'.format(scores.mean()))

sc = StandardScaler()
features = sc.fit_transform(features)

dists = distance.squareform(distance.pdist(features))

#dists[0]

def selectimage(n, m, dists, images):
    image_position = dists[n].argsort()[m]
    image = mh.imread(images[image_position])
    return image

def plotImages(n):
    fig, ax = plt.subplots(1,4, figsize = (15,5))
    
    for i in range(4):
        ax[i].imshow(selectimage(n,i, dists, images))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
    plt.show()
#    fig = plt.gcf()
#    fig.savefig('test.png', dpi=1000)
#    plt.savefig('./test.png')

if __name__ == '__main__':
    plotImage(11)
