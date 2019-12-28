import os
from PIL import Image

def inner(img,m,n):
    a = [-3,-2,-1,0,+1,+2,+3]
    b = [-3,-2,-1,0,+1,+2,+3]
    num = 0
    for i in a:
        for j in b:
            test_m = m + i
            test_n = n + j
            try:
                pixel = img.getpixel((test_m,test_n))
                if pixel[0]>200 and pixel[1]<50 and pixel[2]<50:
                    num += 1
            except:
                pass
    if num > 3:
        return True
    else:
        return False
path ='.\\dataset'
for home, dirs, files in os.walk(path):
    for i in files:
        name = "./New/" + i[:4] + ".jpg"
        i = "./dataset/" + i
        pic = Image.open(i)
        pic = pic.crop((40,10,140,60))
        rows, cols = pic.size
        set_red = []
        set_white = []
        for i in range(rows):
            for j in range(cols):
                pixel = pic.getpixel((i,j))
                if pixel[0]>160 and pixel[1]>160 and pixel[2]>160:
                    pic.putpixel((i,j),(255,255,255))
                if pixel[0]<100 and pixel[1]<100 and pixel[2]<100:
                    if inner(pic,i,j):
                        set_red.append((i,j))  
                    else:
                        set_white.append((i,j))
        for i in set_red:
            pic.putpixel(i,(255,0,0))
        for i in set_white:
            pic.putpixel(i,(255,255,255))
        pic.save(name,quality=100)

#%%
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
import matplotlib.pyplot as plt

model = models.Sequential()
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(250, 250, 3))
conv_base.trainable = False
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(35, activation='softmax'))

train_dir = os.path.join('./train')
validation_dir = os.path.join("./val")
test_dir = os.path.join("./test")
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=25,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=False,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(250, 250),
                                                    batch_size=32,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(250, 250),
                                                        batch_size=32,
                                                        class_mode='categorical')

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=[metrics.mae, metrics.categorical_accuracy])
model.fit_generator(
    train_generator,
    steps_per_epoch=250,
    epochs=35,
    validation_data=validation_generator,
    validation_steps=20
)

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == "block5_conv1":
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

result = model.fit_generator(
    train_generator,
    steps_per_epoch=250,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=30)

acc = result.history['acc']
val_acc = result.history['val_acc']
loss = result.history['loss']
val_loss = result.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label=' Validation accuracy')
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label=' Validation loss')
plt.title("Training and validation loss")
plt.legend()

plt.show()

model.save("classifier.h5")