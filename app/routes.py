from app import app
#from app.ular import *

from flask import url_for, render_template, request

import numpy as np
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K


kapan_hari_ujan = {0: '30 harian lagi', 1: '20 harian lagi', 2: '10 harian lagi'}

size_ = 64

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (size_, size_, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(units = 4096, activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 3, activation = 'softmax'))

model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
#model._make_predict_function()

model.load_weights("kapan_hari_canny_207.h5")


def ini_apa_sih(path):    
    test_image = image.load_img(path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    result = model.predict_classes(test_image)
    
    return kapan_hari_ujan[result[0]]

def factors(num):
  return [x for x in range(1, num+1) if num%x==0]

@app.before_request
def clear_trailing():
    from flask import redirect

    rp = request.path 
    if rp != '/' and rp.endswith('/'):
        return redirect(rp[:-1])

@app.route('/')
def hello_world():
	return render_template('upload_ini.html')
    #return render_template('welcome.html')

@app.route('/hello')
def hello():
    return "Hello, World"

@app.route('/user/<username>')
def show_user_profile(username=None):
    return render_template('user.html', username=username)

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    a = "Post " + str(factors(post_id))
    return render_template(
        'post.html',
        ini = post_id,
        factors = factors(post_id),
        jumlah = len(factors(post_id))
    )
    
@app.route('/hasil_prediksi/<int:post_id>')
def hasil_prediksi(post_id):
    # show the post with the given id, the id is an integer
    a = "Post " + str(factors(post_id))
    return render_template(
        'hasil_prediksi.html',
        ini = post_id,
        factors = factors(post_id),
        jumlah = len(factors(post_id))
    )

@app.route('/path/<path:subpath>')
def show_subpath(subpath):
    # show the subpath after /path/
    return 'Subpath %s' % subpath

@app.route('/login')
def login():
    s = request.args.get('ini')
    s = "tadi minta " + s
    return s

@app.route('/upload_ini')
def upload():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      path = './app/files/' + url_for('upload') + '_' + f.filename
      f.save(path)
      return 'file uploaded successfully'

@app.route('/ini_apa')
def ini_apa():
   return render_template('upload_ini.html')
   
@app.route('/ini_apa_upload', methods = ['GET', 'POST'])
def ini_apa_upload():
   if request.method == 'POST':
      f = request.files['file']
      path = './app/static/images' + url_for('upload') + '_' + f.filename
      path_ = './app/static/images' + url_for('upload') + '__' + f.filename
      path_ini = 'static/images' + url_for('upload') + '_' + f.filename
      f.save(path)
      #return ini_apa(path)
      haha = cv2.imread(path)
      haha = cv2.Canny(haha, 0, 30)
      cv2.imwrite(path_, haha)
      nama_ini = ini_apa_sih(path_)
      return render_template(
        'hasil_prediksi.html',
        ini = nama_ini,
        posisi = path_ini
      )

@app.route('/form')
def exp():
   return render_template('exp.html')

with app.test_request_context():
    print(url_for('hello'))
    print(url_for('show_post', post_id=3))
    print(url_for('static', filename='style.css'))


