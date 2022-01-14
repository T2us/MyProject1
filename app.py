##---------------- Mahotas
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

##---------------- Insatgram Crawling
import insta_download
import urllib.request
#------- 인스타그램 크롤링

import sys
import argparse
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import glob
import os
## json 파일로 출력
import json # <-- json 파일을 위해 확장 라이브러리를 임폴드
from collections import OrderedDict # collections라이브러리에서 OrderedDict함수 불러오기

##---------------- Flask
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

API_URL = 'https://dapi.kakao.com/v2/vision/product/detect'
MYAPP_KEY = '카카오비전 MYAPP KEY를 입력하세요'
global some_queue
some_queue = None
data_list = []

def detect_product(filename):
    headers = {'Authorization': 'KakaoAK {}'.format(MYAPP_KEY)}
    try:
        files = { 'image' : open(filename, 'rb')}
        resp = requests.post(API_URL, headers=headers, files=files)
        resp.status_code

        return resp.json()
    except Exception as e:
        print(str(e))
        sys.exit(0)

def show_products(filename, detection_result):
    try:
        image = Image.open(filename)
    except Exception as e:
        print(str(e))
        sys.exit(0)
    draw = ImageDraw.Draw(image)
    for obj in detection_result['result']['objects']:
        x1 = int(obj['x1']*image.width)
        y1 = int(obj['y1']*image.height)
        x2 = int(obj['x2']*image.width)
        y2 = int(obj['y2']*image.height)
       # print(params)
#        draw.rectangle([(x1,y1), (x2, y2)], fill=None, outline=(255,0,0,255))
#        draw.text((x1+5,y1+5), obj['class'], (255,0,0))
    del draw
    return image


@app.route('/', methods = ['GET', 'POST'])
def main():
#    import os

#    try:
#        [os.remove(f) for f in glob.glob("./temp/*.png")]
#        [os.remove(f) for f in glob.glob("./static/cropPic/*.jpg")]
#        [os.remove(f) for f in glob.glob("./static/cropImg/*.jpg")]
#        os.remove("./pic/000.jpg")
#    except Exception as e:
#        print("이미지 제거 실패")
    
    return render_template('main.html')

@app.route('/main')
def _quit():
    os._exit(0)
    return render_template('main.html')

@app.route('/pic', methods = ['GET', 'POST'])
def render_file():
    return render_template('upload.html')

@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
#    if request.method == 'POST':
#        f = request.files['file']
#        f.save('C:/inetpub/flask/pic/000.jpg')

    files=glob.glob('./pic/*.jpg')+glob.glob('./pic/*.jpeg')+glob.glob('./pic/*.png')
    count = 0
    for i in files:
        head,tail = os.path.split(i)
        detection_result = detect_product(i)
        image = show_products(i, detection_result)
        temp = detection_result.get("result").get("objects")
        temp2 = detection_result.get("result")
        
        for j in temp:
            try:
                if j['score'] >= 0.88:
                    fileName = format(count, '03')
                    cropping_area = (j['x1']*temp2.get("width"), j['y1']*temp2.get("height"), j['x2']*temp2.get("width"), j['y2']*temp2.get("height"))
                    croppedImage=image.crop(cropping_area)
                    croppedImage.save('./static/cropPic/%s' % fileName +'.jpg')
                    path="C:/inetpub/flask/static/cropPic"
                    files = os.listdir(path)
                    count = count + 1
            except Exception as e:
                print("error")
    return render_template('picupload.html', pics=files)

@app.route('/fileview', methods = ['GET', 'POST'])
def f_View():
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
    import json
    import shutil
    #filename = requests.getParameter['filename']
    filename = request.args.get('filename', "test")
    dict = { 'filename' : filename }
    with open('Test.json', 'w', encoding='utf-8') as make_f:
        json.dump(dict, make_f, indent="\t")

    with open('./Test.json', 'rt', encoding="utf-8-sig") as f:
        config = json.load(f)
    fn = config['filename'].replace("'","")
    shutil.copyfile("./static/cropPic/"+fn, "./img/000.jpg")
    global images
    images = glob('./img/*.jpg')
    #images = glob('./img/*.jpg')
    features = []
    labels = []

    images[:5]

    images[0][19:-len('.jpg')]

    im = mh.imread(images[0])
    im = mh.colors.rgb2gray(im, dtype=np.uint8)
    im

    mh.features.haralick(im)



    start = time.time()

    for im in images:
        labels.append(im[19:-len('.jpg')])
        im = mh.imread(im)
        im = mh.colors.rgb2gray(im, dtype = np.uint8)
        features.append(mh.features.haralick(im).ravel())

    print(f'fit time : {time.time() - start}')

    features = np.array(features)
    labels = np.array(labels)

    clf = Pipeline([('preproc', StandardScaler()),('classifier', LogisticRegression())])

    sc = StandardScaler()
    features = sc.fit_transform(features)
    global dists
    dists = distance.squareform(distance.pdist(features))
    
    global INAME
    INAME = []

    plotImages(0)

    with open('shop_link_fileview.json', 'rt', encoding="utf-8-sig") as f:
        linkValue = json.load(f)
    time.sleep(15)
    return render_template('view.html', one=linkValue[str(INAME[1])], two=linkValue[str(INAME[2])], three=linkValue[str(INAME[3])])


@app.route('/insta', methods = ['GET', 'POST'])
def insta():
    import time
    myimg = insta_download.get_img_url()
    if myimg['img']:
        num = 0
        imgs = list(set(myimg['img']))

    for img in imgs:
        urllib.request.urlretrieve(img,'./temp/temp_image'+str(num)+'.png')
        num += 1
        
    if myimg['video']:
        num = 0

    files=glob.glob('./temp/*.jpg')+glob.glob('./temp/*.jpeg')+glob.glob('./temp/*.png')
    count = 0
    for i in files:
        head,tail = os.path.split(i)
        detection_result = detect_product(i)
        image = show_products(i, detection_result)
        temp = detection_result.get("result").get("objects")
        temp2 = detection_result.get("result")
        
        for j in temp:
            try:
                fileName = format(count, '03')
                cropping_area = (j['x1']*temp2.get("width"), j['y1']*temp2.get("height"), j['x2']*temp2.get("width"), j['y2']*temp2.get("height"))
                croppedImage=image.crop(cropping_area)
                croppedImage.save('./static/cropimg/%s' % fileName +'.jpg')
                path="C:/inetpub/flask/static/cropimg"
                files = os.listdir(path)
                count = count + 1
            except Exception as e:
                print("error")
    return render_template('insta.html', pics=files)

@app.route('/jsontest', methods = ['GET', 'POST'])
def json():
    return render_template('jsontest.html')

@app.route('/post', methods = ['POST'])
def post():
    import json
    
    value = request.form['input']
    wishN = request.form['wish']
    msg = "Insta Nickname = %s" %value
    dict = { 'keyword' : value, 'wish_num' : int(wishN), 'cd_path' : "./chromedriver.exe" }
    with open('MyJson.json', 'w', encoding='utf-8') as m_f:
        json.dump(dict, m_f, indent="\t")
    return render_template('post.html')

@app.route('/view', methods = ['GET', 'POST'])
def view():
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
    import json
    import shutil
    #filename = requests.getParameter['filename']
    filename = request.args.get('filename', "test")
    dict = { 'filename' : filename }
    with open('Test.json', 'w', encoding='utf-8') as make_f:
        json.dump(dict, make_f, indent="\t")

    with open('./Test.json', 'rt', encoding="utf-8-sig") as f:
        config = json.load(f)
    fn = config['filename'].replace("'","")
    shutil.copyfile("./static/cropimg/"+fn, "./findimg/000.jpg")
    global images
    images = glob('./findimg/*.jpg')
    #images = glob('./img/*.jpg')
    #images = glob('C:/Users/1234/Desktop/test/img')
    features = []
    labels = []

    images[:5]

    images[0][19:-len('.jpg')]

    im = mh.imread(images[0])
    im = mh.colors.rgb2gray(im, dtype=np.uint8)
    im

    mh.features.haralick(im)



    start = time.time()

    for im in images:
        labels.append(im[19:-len('.jpg')])
        im = mh.imread(im)
        im = mh.colors.rgb2gray(im, dtype = np.uint8)
        features.append(mh.features.haralick(im).ravel())

    print(f'fit time : {time.time() - start}')

    features = np.array(features)
    labels = np.array(labels)

    clf = Pipeline([('preproc', StandardScaler()),('classifier', LogisticRegression())])

    sc = StandardScaler()
    features = sc.fit_transform(features)
    global dists
    dists = distance.squareform(distance.pdist(features))
    global INAME
    INAME = []

    plotImages(0)

    with open('shop_link_real.json', 'rt', encoding="utf-8-sig") as f:
        linkValue = json.load(f)

    time.sleep(15)
    return render_template('view.html', one=linkValue[str(INAME[1])], two=linkValue[str(INAME[2])], three=linkValue[str(INAME[3])])

def selectImage( n, m, dists, images):
    image_position = dists[n].argsort()[m]
    image = mh.imread( images[image_position] )
    return image

def plotImages(n):
    plt.rcParams['figure.figsize'] = (9, 9)

    plt.subplot(4, 1, 1)
    plt.imshow(selectImage(n, 0, dists, images))
    plt.title('Original')
    print(dists[n].argsort()[0])
    INAME.append(dists[n].argsort()[0])
    plt.xticks([])
    plt.yticks([])
    
    #plt.subplot(2, 1)
    plt.subplot(4, 1, 2)
    plt.imshow(selectImage(n, 1, dists, images))
    plt.title('1st similar one')
    print(dists[n].argsort()[1])
    INAME.append((dists[n].argsort()[1]))
    plt.xticks([])
    plt.yticks([])
    
    #plt.subplot(3, 1)
    plt.subplot(4, 1, 3)
    plt.imshow(selectImage(n, 2, dists, images))
    plt.title('2nd similar one')
    print(dists[n].argsort()[2])
    INAME.append((dists[n].argsort()[2]))
    plt.xticks([])
    plt.yticks([])
    
    #plt.subplot(4, 1)
    plt.subplot(4, 1, 4)
    plt.imshow(selectImage(n, 3, dists, images))
    plt.title('3rd similar one')
    print(dists[n].argsort()[3])
    INAME.append((dists[n].argsort()[3]))
    plt.xticks([])
    plt.yticks([])
    print(INAME)
    plt.savefig('./static/result/result.png', bbox_inches='tight')

def start_flaskapp(queue):
   some_queue = queue
   app.run(app.run(host='0.0.0.0'))

if __name__ == '__main__':
    
    app.run(host='0.0.0.0')
        
    
