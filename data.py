import os ,shutil
orginal_dataset_dir ="C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\dataset"
os.lstat(orginal_dataset_dir)

base_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplerning\\cats_and_dogs"



# kedi için test train validation veri seçimleri

train_cats_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\base_dir\\train\\cats"
o_train_cats_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\\dataset\\train1\\train"
fnames= ["cat.{}.jpg".format(i)  for i in range(1000)]
for fname in fnames:
    src = os.path.join(o_train_cats_dir,fname)
    dst = os.path.join(train_cats_dir,fname)
    print(dst)
    shutil.copyfile(src,dst)


validation_cats_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\base_dir\\validation\\cats"
o_validation_cats_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\\dataset\\train1\\train"
fnames= ["cat.{}.jpg".format(i)  for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(o_train_cats_dir,fname)
    dst = os.path.join(validation_cats_dir,fname)
    print(dst)
    shutil.copyfile(src,dst)



validation_cats_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\base_dir\\test\\cats"
o_validation_cats_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\\dataset\\train1\\train"
fnames= ["cat.{}.jpg".format(i)  for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(o_train_cats_dir,fname)
    dst = os.path.join(validation_cats_dir,fname)
    print(dst)
    shutil.copyfile(src,dst)


    

# köpek için test train validation veri seçimleri

train_cats_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\base_dir\\train\\dogs"
o_train_cats_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\\dataset\\train1\\train"
fnames= ["dog.{}.jpg".format(i)  for i in range(1000)]
for fname in fnames:
    src = os.path.join(o_train_cats_dir,fname)
    dst = os.path.join(train_cats_dir,fname)
    print(dst)
    shutil.copyfile(src,dst)


validation_cats_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\base_dir\\validation\\dogs"
o_validation_cats_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\\dataset\\train1\\train"
fnames= ["dog.{}.jpg".format(i)  for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(o_train_cats_dir,fname)
    dst = os.path.join(validation_cats_dir,fname)
    print(dst)
    shutil.copyfile(src,dst)



validation_cats_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\base_dir\\test\\dogs"
o_validation_cats_dir = "C:\\Users\\Maşallah\\Desktop\\Deeplearning\\cats_and_dogs\\\dataset\\train1\\train"
fnames= ["dog.{}.jpg".format(i)  for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(o_train_cats_dir,fname)
    dst = os.path.join(validation_cats_dir,fname)
    print(dst)
    shutil.copyfile(src,dst)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()