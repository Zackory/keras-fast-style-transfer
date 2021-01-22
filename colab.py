import os, sys, glob, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2

# Needed for download Keras pretrained models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Add
from keras.optimizers import Adam, SGD
from keras_applications import vgg19
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

checkpoint_dir = 'models'
style_image_name = 'udnie.jpg'
test_image_name = 'stata.jpg'
test_dir = 'tests'
checkpoint_iterations = 100
epochs = 2
batch_size = 5

content_weight = 7.5
# content_weight = 20.0
# style_weight = 200.0
style_weight = 50.0
tv_weight = 200.0
lr = 0.001

train_path = os.path.join('data', 'train2014')
# train_path = 'train2014'
debug = True

def save_image(out_path, image):
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, image)

def get_image(src, image_size=False):
    image = cv2.imread(src)
    if not (len(image.shape) == 3 and image.shape[2] == 3):
        image = np.dstack((image, image, image))
    if image_size != False:
        image = cv2.resize(image, image_size[:2])
    return image

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

makedirs(checkpoint_dir)
makedirs(test_dir)

style_target = get_image(style_image_name)
content_targets = glob.glob(os.path.join(train_path, '*.jpg'))
content_targets = content_targets[:-(len(content_targets) % batch_size)]
print('Found %d images' % len(content_targets))

batch_shape = (batch_size,256,256,3)
style_shape = (1,) + style_target.shape
print('Style shape:', style_shape)
print('Batch shape:', batch_shape)

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layer = 'block4_conv2'
# style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1']
# content_layer = 'block2_conv2'
model = vgg19.VGG19(weights='imagenet', include_top=False, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
preprocess_input = vgg19.preprocess_input
# Remove the last pooling layer
model = Model(model.input, model.layers[-2].output)
model.summary()

# x = Conv2D(32, 9, strides=(1, 1), padding='same', activation='relu')(model.input)
# x = Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2D(128, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2DTranspose(32, 3, strides=(2, 2), padding='same', activation='relu')(x)
# transform_output = Conv2D(3, 9, strides=(1, 1), padding='same', activation='sigmoid')(x)# * 150 + 255.0/2
# # transform_output = Conv2D(3, 9, strides=(1, 1), padding='same', activation='tanh')(x)# * 150 + 255.0/2
# print(model.input, transform_output)
# transform_model = Model(model.input, transform_output)
# transform_model.summary()

x = Conv2D(32, 9, strides=(1, 1), padding='same', activation='relu')(model.input)
x = Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
x = Conv2D(128, 3, strides=(2, 2), padding='same', activation='relu')(x)
# Residual blocks
x = Add()([x, Conv2D(128, 3, strides=(1, 1), padding='same', activation='linear')(Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(x))])
x = Add()([x, Conv2D(128, 3, strides=(1, 1), padding='same', activation='linear')(Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(x))])
x = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
x = Conv2DTranspose(32, 3, strides=(2, 2), padding='same', activation='relu')(x)
transform_output = Conv2D(3, 9, strides=(1, 1), padding='same', activation='sigmoid')(x)# * 150 + 255.0/2
print(model.input, transform_output)
transform_model = Model(model.input, transform_output)
transform_model.summary()

style_features = {}

style_pre = K.constant(np.array([style_target]))
style_pre = preprocess_input(style_pre, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
for layer in style_layers:
    features = Model(inputs=model.input, outputs=model.get_layer(layer).output).predict(style_pre, steps=1)
    features = np.reshape(features, (-1, features.shape[3]))
    style_features[layer] = K.constant(np.matmul(features.T, features) / features.size)
    # features = K.reshape(features, (-1, features.shape[3]))
    # style_features[layer] = K.constant(K.dot(K.transpose(features), features) / tf.size(features, out_type=tf.dtypes.float32))

content_input = K.placeholder(shape=batch_shape)
test_input = K.placeholder(shape=(None, None, None, 3))

preds = transform_model(content_input / 255.0) * 255.0
test_preds = transform_model(test_input / 255.0) * 255.0
# preds = transform_model(content_input / 255.0) * 150 + 255./2
# test_preds = transform_model(test_input / 255.0) * 150 + 255./2
preds_pre = preprocess_input(preds, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)

l2_loss = lambda x: K.sum(K.square(x))

content_model = Model(inputs=model.input, outputs=model.get_layer(content_layer).output)
pred_output = content_model(preds_pre)
content_pre = preprocess_input(content_input, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
content_output = content_model(content_pre)
content_size = K.cast(K.prod(K.shape(content_output)), 'float32')
content_loss = content_weight * 2 * l2_loss(pred_output - content_output) / content_size

style_loss = 0
for layer in style_layers:
    layer_output = Model(inputs=model.input, outputs=model.get_layer(layer).output)(preds_pre)
    if len(layer_output.shape) == 4:
        bs, height, width, filters = layer_output.shape
    else:
        bs, height, width, filters = K.shape(layer_output)
    # size = K.cast(bs * height * width * filters, 'float32')
    size = K.cast(height * width * filters, 'float32')
    feats = K.reshape(layer_output, (bs * width * height, filters))
    grams = K.dot(K.transpose(feats), feats) / size
    # feats = K.reshape(layer_output, (bs, width * height, filters))
    # feats_T = tf.transpose(a=feats, perm=[0,2,1])
    # grams = tf.matmul(feats_T, feats) / size
    # feats_T = K.permute_dimensions(feats, (0,2,1))
    # grams = K.batch_dot(feats_T, feats) / size
    style_gram = style_features[layer]
    style_gram_size = K.cast(K.prod(K.shape(style_gram)), 'float32')
    # style_loss += l2_loss(grams - style_gram) / style_gram_size
    style_loss += l2_loss(grams - style_gram) / style_gram_size
style_loss = style_weight * 2 * style_loss / batch_size

print(K.shape(preds)[0])
# y1, y2, y3, y4 = K.shape(preds)
y1 = K.shape(preds)[0]
y2 = K.shape(preds)[1]
y3 = K.shape(preds)[2]
y4 = K.shape(preds)[3]
tv_y_size = K.cast(y1*(y2-1)*y3*y4, 'float32')
tv_x_size = K.cast(y1*y2*(y3-1)*y4, 'float32')
y_tv = l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
x_tv = l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

loss = content_loss + style_loss + tv_loss

adam = Adam(lr=lr)
# sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
param_updates = adam.get_updates(params=transform_model.trainable_weights, loss=loss)
train_batch = K.function(inputs=[K.learning_phase(), content_input], outputs=[loss, content_loss, style_loss, tv_loss], updates=param_updates)
test_batch = K.function(inputs=[K.learning_phase(), test_input], outputs=[test_preds])

param_updates_content = adam.get_updates(params=transform_model.trainable_weights, loss=content_loss)
train_batch_content = K.function(inputs=[K.learning_phase(), content_input], outputs=[content_loss], updates=param_updates_content)

X_test = np.array([get_image(test_image_name)])
print(np.min(X_test), np.max(X_test))
print('Test image size:', np.shape(X_test))
print('Beginning!', len(content_targets) // batch_size, 'batches per epoch')
train_phase = 1
test_phase = 0
begin_time = time.time()

# TODO: First train transform model with only content loss to learn the identity mapping, then fine tune with the full loss function
# for iteration in range(1000):
#     start_time = time.time()
#     curr = iteration * batch_size
#     step = curr + batch_size
#     X_batch = np.zeros(batch_shape, dtype=np.float32)
#     for j, image in enumerate(content_targets[curr:step]):
#         X_batch[j] = get_image(image, (256,256,3)).astype(np.float32)
#     c_loss = train_batch_content([train_phase, X_batch])[0]
#     # TODO: Save checkpoint and test image every args.checkpoint_iterations
#     if iteration % checkpoint_iterations == 0:
#         if debug:
#             print('Iteration:', iteration, 'Batch time: %.2f' % (time.time() - start_time), 'total time: %.2f' % (time.time() - begin_time), 'content:', c_loss)
#         t = time.time()
#         transform_model.save(os.path.join(checkpoint_dir, os.path.splitext(os.path.basename(style_image_name))[0] + ('_content_%d.h5' % iteration)))
#         test_image = test_batch([test_phase, X_test])[0][0]# * 255.0
#         test_image = test_image.astype(int)
#         save_image(os.path.join(test_dir, os.path.splitext(os.path.basename(test_image_name))[0] + ('_content_%d.jpg' % iteration)), test_image)
#         print('Save test image time:', time.time() - t)


for epoch in range(epochs):
    for iteration in range(len(content_targets) // batch_size):
        start_time = time.time()
        curr = iteration * batch_size
        step = curr + batch_size
        X_batch = np.zeros(batch_shape, dtype=np.float32)
        for j, image in enumerate(content_targets[curr:step]):
            X_batch[j] = get_image(image, (256,256,3)).astype(np.float32)
        batch_loss, c_loss, s_loss, t_loss = train_batch([train_phase, X_batch])
        if debug:
            print('Iteration:', (epoch+1)*iteration, 'Batch time: %.2f' % (time.time() - start_time), 'total time: %.2f' % (time.time() - begin_time), 'content:', c_loss, 'style:', s_loss, 'tv:', t_loss)
        if iteration % checkpoint_iterations == 0:
            t = time.time()
            transform_model.save(os.path.join(checkpoint_dir, os.path.splitext(os.path.basename(style_image_name))[0] + ('_%d.h5' % ((epoch+1)*iteration))))
            test_image = test_batch([test_phase, X_test])[0][0]# * 255.0
            test_image = test_image.astype(int)
            save_image(os.path.join(test_dir, os.path.splitext(os.path.basename(test_image_name))[0] + ('_%d.jpg' % ((epoch+1)*iteration))), test_image)
            print('Save test image time:', time.time() - t)

