from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import regex
import sys

#np.set_printoptions(threshold=sys.maxsize)
np.random.seed(1024)

train_path = "OCR/Data Preprocessing/archive/CASIA-HWDB_Train/Train" 
test_path = "OCR/Data Preprocessing/archive/CASIA-HWDB_Test/Test" 
train_chinese = os.listdir(train_path)
test_chinese = os.listdir(test_path)

chinese = "大家准备开始女台"

df_train = pd.DataFrame(columns=["chinese","filename"])
for i in chinese:
    if (regex.findall(r'\p{Han}+', i)) == []: continue
    print(i)
    l = os.listdir(train_path+'/'+i)
    df_train = pd.concat([df_train, pd.DataFrame({
        "chinese":i,
        "filename":l
    })],ignore_index = True)
    
df_train["filename"] = df_train['chinese']+"/"+df_train["filename"]

def img_map(image):
    # convert images to grayscale
    image = image.convert('L')
    # convert it to standard size 28*28 same as mnist
    image = image.resize((28,28), Image.LANCZOS)
    # increase contrast
    enh_col = ImageEnhance.Contrast(image)
    factor = np.random.uniform(2,3,1)
    image = enh_col.enhance(factor=factor)
    
    return image

# map image to array
def to_array(image):
    return np.array(image)

# rotate image to generate more datapoints
def img_rotate_CCW(image):
    r = np.random.uniform(1,3,1)
    return image.rotate(r, fillcolor="white")

def img_rotate_CW(image):
    r = np.random.uniform(1,3,1)
    return image.rotate(360-r, fillcolor="white")

def img_trans(imgs, direction):
    s = imgs.shape
    img1 = np.zeros(shape=s)+255
    px = np.random.choice([1,2,3],1)[0]
    if direction == "left":
        img1[:,:,:s[2]-px] = imgs[:,:,px:]
    if direction == "right":
        img1[:,:,px:] = imgs[:,:,:s[2]-px]
    if direction == "up":
        img1[:,:s[1]-px,:] = imgs[:,px:,:]
    if direction == "down":
        img1[:,px:,:] = imgs[:,:s[1]-px,:]
    return img1

train_images = map(Image.open,train_path+"/"+df_train["filename"])

train_img = list(map(img_map,train_images))

train_images_CW = map(img_rotate_CW,train_img) #顺时针2度
train_images_CCW = map(img_rotate_CCW,train_img) #逆时针2度

train_images_CW = map(to_array,train_images_CW)
train_images_CCW = map(to_array,train_images_CCW)
train_images = map(to_array,train_img)

train_images_CW = np.array(list(train_images_CW))
train_images_CCW = np.array(list(train_images_CCW))
train_images = np.array(list(train_images))

train_images_left = img_trans(train_images,"left")
train_images_right = img_trans(train_images,"right")
train_images_up = img_trans(train_images,"up")
train_images_down = img_trans(train_images,"down")

i = np.random.choice(list(range(1,train_images.shape[0])))
fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(1,7)
ax1.imshow(train_images[i,:,:],cmap='gray')
ax1.set_xticks([])
ax1.set_yticks([])
ax2.imshow(train_images_left[i,:,:],cmap='gray')
ax2.set_xticks([])
ax2.set_yticks([])
ax3.imshow(train_images_right[i,:,:],cmap='gray')
ax3.set_xticks([])
ax3.set_yticks([])
ax4.imshow(train_images_up[i,:,:],cmap='gray')
ax4.set_xticks([])
ax4.set_yticks([])
ax5.imshow(train_images_down[i,:,:],cmap='gray')
ax5.set_xticks([])
ax5.set_yticks([])
ax6.imshow(train_images_CW[i,:,:],cmap='gray')
ax6.set_xticks([])
ax6.set_yticks([])
ax7.imshow(train_images_CCW[i,:,:],cmap='gray')
ax7.set_xticks([])
ax7.set_yticks([])

train_images = np.concatenate((train_images,train_images_left,
                               train_images_right,train_images_up,
                               train_images_down,train_images_CW,
                              train_images_CCW))
x = np.array(df_train["chinese"])
df1 = pd.DataFrame(columns=["chinese"])
y_train = np.tile(x,7)

df_test = pd.DataFrame(columns=["chinese","filename"])
for i in chinese:
    l = os.listdir(test_path+'/'+i)
    df_test = pd.concat([df_test, pd.DataFrame({
        "chinese":i,
        "filename":l
    })],ignore_index = True)
    
df_test["filename"] = df_test['chinese']+"/"+df_test["filename"]

test_images = map(Image.open,test_path+"/"+df_test["filename"])

test_img = list(map(img_map,test_images))

test_images_CW = map(img_rotate_CW,test_img) #顺时针2度
test_images_CCW = map(img_rotate_CCW,test_img) #逆时针2度

test_images_CW = map(to_array,test_images_CW)
test_images_CCW = map(to_array,test_images_CCW)
test_images = map(to_array,test_img)

test_images_CW = np.array(list(test_images_CW))
test_images_CCW = np.array(list(test_images_CCW))
test_images = np.array(list(test_images))

test_images_left = img_trans(test_images,"left")
test_images_right = img_trans(test_images,"right")
test_images_up = img_trans(test_images,"up")
test_images_down = img_trans(test_images,"down")

test_images = np.concatenate((test_images,test_images_left,
                               test_images_right,test_images_up,
                               test_images_down,test_images_CW,
                             test_images_CCW))
x = np.array(df_test["chinese"])
df1 = pd.DataFrame(columns=["chinese"])
y_test = np.tile(x,7)

train_idx = list(range(0,len(y_train)))
np.random.shuffle(train_idx)
np.random.shuffle(train_idx)
x_train = train_images[train_idx]
y_train = y_train[train_idx]

test_idx = list(range(0,len(y_test)))
np.random.shuffle(test_idx)
np.random.shuffle(test_idx)
x_test = test_images[test_idx]
y_test = y_test[test_idx]

np.savez("data.npz",
        x_train = x_train,
        y_train = y_train,
        x_test = x_test,
        y_test = y_test)

#print(y_train)

#plt.show()