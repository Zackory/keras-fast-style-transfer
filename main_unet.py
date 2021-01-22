import os, sys, glob, time, argparse, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from utils import save_image, get_image, makedirs

# Needed for download Keras pretrained models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Add
from keras.optimizers import Adam, SGD
from keras_applications import resnet50, vgg19, nasnet, mobilenet_v2
import tensorflow as tf

# from classification_models.keras import Classifiers

import segmentation_models as sm

train_path = os.path.join('data', 'train2014')
debug = True

parser = argparse.ArgumentParser(description='Keras Fast Style Transfer')
parser.add_argument('--checkpoint-dir', default='checkpoints', help='Directory to save checkpoints in', required=True)
parser.add_argument('--style', help='Style image path', required=True)
parser.add_argument('--test', default=False, help='Test image path')
parser.add_argument('--test-dir', default=False, help='Test image save directory')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=5, help='Batch size')
parser.add_argument('--checkpoint-iterations', type=int, default=20, help='Checkpoint frequency (in batches)')

# For VGG19
parser.add_argument('--content-weight', type=float, default=15.0, help='Content weight')
parser.add_argument('--style-weight', type=float, default=200.0, help='Style weight')
# Default weights
# parser.add_argument('--content-weight', type=float, default=15.0, help='Content weight')
# parser.add_argument('--style-weight', type=float, default=100.0, help='Style weight')
parser.add_argument('--tv-weight', type=float, default=200.0, help='Total variation regularization weight')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

makedirs(args.checkpoint_dir)
makedirs(args.test_dir)
epochs = args.epochs
batch_size = args.batch_size

style_target = get_image(args.style)
content_targets = glob.glob(os.path.join(train_path, '*.jpg'))
content_targets = content_targets[:-(len(content_targets) % batch_size)]
print('Found %d images' % len(content_targets))

batch_shape = (batch_size,256,256,3)
style_shape = (1,) + style_target.shape
print('Style shape:', style_shape)
print('Batch shape:', batch_shape)

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
# style_layers = ['block1_conv1', 'block3_conv1', 'block5_conv1']
content_layer = 'block4_conv2'
model = vgg19.VGG19(weights='imagenet', include_top=False, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)

model = Model(model.input, model.layers[-2].output)
model.summary()

BACKBONE = 'efficientnetb0'#'resnet18'
preprocess_input = sm.get_preprocessing(BACKBONE)
transform_model = sm.Unet(BACKBONE, encoder_weights=None, activation='sigmoid', classes=3)#, encoder_weights='imagenet')
transform_model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
transform_model.summary()
# exit()

# x = Conv2D(32, 9, strides=(1, 1), padding='same', activation='relu')(model.input)
# x = Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2D(128, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2D(256, 3, strides=(2, 2), padding='same', activation='relu')(x)
#  # Residual blocks
# x = Add()([x, Conv2D(256, 3, strides=(1, 1), padding='same', activation='linear')(Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu')(x))])
# x = Add()([x, Conv2D(256, 3, strides=(1, 1), padding='same', activation='linear')(Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu')(x))])
# x = Add()([x, Conv2D(256, 3, strides=(1, 1), padding='same', activation='linear')(Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu')(x))])
# x = Add()([x, Conv2D(256, 3, strides=(1, 1), padding='same', activation='linear')(Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu')(x))])
# x = Add()([x, Conv2D(256, 3, strides=(1, 1), padding='same', activation='linear')(Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu')(x))])
# x = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2DTranspose(32, 3, strides=(2, 2), padding='same', activation='relu')(x)
# transform_output = Conv2D(3, 9, strides=(1, 1), padding='same', activation='sigmoid')(x)# * 150 + 255.0/2
# print(model.input, transform_output)
# transform_model = Model(model.input, transform_output)
# transform_model.summary()
# exit()


style_features = {}

style_pre = np.array([style_target])
style_pre = vgg19.preprocess_input(style_pre, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
# style_pre = preprocess_input(style_pre)
for layer in style_layers:
    features = Model(inputs=model.input, outputs=model.get_layer(layer).output).predict(style_pre)
    features = np.reshape(features, (-1, features.shape[3]))
    style_features[layer] = K.constant(np.matmul(features.T, features) / features.size)


# content_input = K.placeholder(shape=model.layers[0].input_shape)
content_input = K.placeholder(shape=batch_shape)
test_input = K.placeholder(shape=(None, None, None, 3))

# preds = transform_model(content_input / 255.0) * 255.0
# test_preds = transform_model(test_input / 255.0) * 255.0
preds = transform_model(content_input) * 255.0
test_preds = transform_model(test_input) * 255.0
preds_pre = vgg19.preprocess_input(preds, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
# preds_pre = preprocess_input(preds)

l2_loss = lambda x: K.sum(K.square(x))

content_model = Model(inputs=model.input, outputs=model.get_layer(content_layer).output)
pred_output = content_model(preds_pre)
content_pre = vgg19.preprocess_input(content_input, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
content_output = content_model(content_pre)
content_size = K.cast(K.prod(K.shape(content_output)), 'float32')
content_loss = args.content_weight * l2_loss(pred_output - content_output) / content_size

# style_losses = []
style_loss = 0
for layer in style_layers:
    layer_output = Model(inputs=model.input, outputs=model.get_layer(layer).output)(preds_pre)
    if len(layer_output.shape) == 4:
        bs, height, width, filters = layer_output.shape
    else:
        bs, height, width, filters = K.shape(layer_output)
    size = K.cast(bs * height * width * filters, 'float32')
    feats = K.reshape(layer_output, (bs * width * height, filters))
    grams = K.dot(K.transpose(feats), feats) / size
    style_gram = style_features[layer]
    style_gram_size = K.cast(K.prod(K.shape(style_gram)), 'float32')
    style_loss += l2_loss(grams - style_gram) / style_gram_size
style_loss = args.style_weight * style_loss / batch_size

y1, y2, y3, y4 = K.shape(preds)
tv_y_size = K.cast(y1*(y2-1)*y3*y4, 'float32')
tv_x_size = K.cast(y1*y2*(y3-1)*y4, 'float32')
y_tv = l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
x_tv = l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
tv_loss = args.tv_weight*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

loss = content_loss + style_loss + tv_loss
# loss = content_loss + style_loss

adam = Adam(lr=args.lr)
# sgd = SGD(lr=args.lr, momentum=0.9, nesterov=True)
param_updates = adam.get_updates(params=transform_model.trainable_weights, loss=loss)
train_batch = K.function(inputs=[K.learning_phase(), content_input], outputs=[loss, content_loss, style_loss, tv_loss], updates=param_updates)
# train_batch = K.function(inputs=[K.learning_phase(), content_input], outputs=[loss, content_loss, style_loss], updates=param_updates)
test_batch = K.function(inputs=[K.learning_phase(), test_input], outputs=[test_preds])

X_test = np.array([get_image(args.test)])
X_test = preprocess_input(X_test)
print(np.min(X_test), np.max(X_test))
print('Test image size:', np.shape(X_test))
print('Beginning!', len(content_targets) // batch_size, 'batches per epoch')
train_phase = 1
test_phase = 0
begin_time = time.time()
for epoch in range(epochs):
    for iteration in range(len(content_targets) // batch_size):
        start_time = time.time()
        curr = iteration * batch_size
        step = curr + batch_size
        X_batch = np.zeros(batch_shape, dtype=np.float32)
        for j, image in enumerate(content_targets[curr:step]):
            X_batch[j] = get_image(image, (256,256,3)).astype(np.float32)
        X_batch = preprocess_input(X_batch)
        batch_loss, c_loss, s_loss, t_loss = train_batch([train_phase, X_batch])
        # batch_loss, c_loss, s_loss = train_batch([train_phase, X_batch])
        if debug:
            print('Batch time: %.2f' % (time.time() - start_time), 'total time: %.2f' % (time.time() - begin_time), 'content:', c_loss, 'style:', s_loss, 'tv:', t_loss)
            # print('Batch time: %.2f' % (time.time() - start_time), 'total time: %.2f' % (time.time() - begin_time), 'content:', c_loss, 'style:', s_loss)
        # TODO: Save checkpoint and test image every args.checkpoint_iterations
        if iteration % args.checkpoint_iterations == 0:
            t = time.time()
            transform_model.save(os.path.join(args.checkpoint_dir, os.path.splitext(os.path.basename(args.style))[0] + ('_%d.h5' % iteration)))
            test_image = test_batch([test_phase, X_test])[0][0]# * 255.0
            test_image = test_image.astype(int)
            # print(np.min(test_image), np.max(test_image))
            save_image(os.path.join(args.test_dir, os.path.splitext(os.path.basename(args.test))[0] + ('_%d.jpg' % iteration)), test_image)
            print('Save test image time:', time.time() - t)
