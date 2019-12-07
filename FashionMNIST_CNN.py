#!/usr/bin/env python
# coding: utf-8
# [Keras Dataset](https://keras.io/ko/datasets/#-mnist)

# In[1]:


# 1. 데이터 불러오기

from keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[2]:


# 2. 이미지 데이터 확인하기 🖼

import matplotlib.pyplot as plt

image = X_train[0]

plt.imshow(image, cmap = plt.cm.gray)


# In[3]:


# 3-1. 이미지 데이터 전처리 : 2차원->3차원 🌟🌟🌟

w = h = 28

X_train = X_train.reshape(X_train.shape[0],w,h,1)
X_test = X_test.reshape(X_test.shape[0],w,h,1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[4]:


# 3-2. 이미지 데이터 전처리 : Normalzation 

X_train = X_train/255
X_test = X_test/255


# In[5]:


# 4. Label categorical (one-hot encoding) 

from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[6]:


# 5. 모델 생성 : CNN

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5),
                 input_shape=(28,28,1),
                 padding='same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(5,5),
                 input_shape=(28,28,1),
                 padding='same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.25)
          model.add(Dense(10,activation='softmax'))

print(model.summary())


# In[12]:


# 6. Compile - Optimizer, Loss function 설정

model.compile(loss='categorical_crossentropy', 
              optimizer='sgd',
              metrics=['accuracy'])


# In[13]:


#validation set

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=0.2, 
                                                  random_state=9)

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)


# In[21]:


# 7. 모델 학습시키기

batch_size = 64
epochs = 15

history = model.fit(X_train, y_train, 
          batch_size=batch_size, 
          epochs=epochs, 
          validation_data=(X_val, y_val), 
          shuffle=True, 
          verbose=1)


# In[22]:


# 8. 모델 평가하기

test_loss, test_acc = model.evaluate(X_test, y_test)

print(test_loss, test_acc)


# In[23]:


# 9. 이미지를 랜덤으로 선택해 훈련된 모델로 예측 🖼

import numpy
for index in numpy.random.choice(len(y_test), 3, replace = False):
    predicted = model.predict(X_test[index:index + 1])[0]
    label = y_test[index]
    result_label = numpy.where(label == numpy.amax(label))
    result_predicted = numpy.where(predicted == numpy.amax(predicted))
    title = "Label value = %s  Predicted value = %s " % (result_label[0], result_predicted[0])
    
    fig = plt.figure(1, figsize = (3,3))
    ax1 = fig.add_axes((0,0,.8,.8))
    ax1.set_title(title)
    images = X_test
    plt.imshow(images[index].reshape(28, 28), cmap = 'Greys', interpolation = 'nearest')
    plt.show()


# In[24]:


# 10. 학습 시각화하기

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('epoch')
plt.xlabel('accuracy')
plt.legend(['train','test'],loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('epoch')
plt.xlabel('loss')
plt.legend(['train','test'],loc='upper left')
plt.show()


# In[ ]:




