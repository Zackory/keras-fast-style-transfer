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
# parser.add_argument('--content-weight', type=float, default=2.5, help='Content weight')
# parser.add_argument('--style-weight', type=float, default=200.0, help='Style weight')
# parser.add_argument('--tv-weight', type=float, default=500.0, help='Total variation regularization weight')
# Working, lr=0.0025
# parser.add_argument('--content-weight', type=float, default=10, help='Content weight')
# parser.add_argument('--style-weight', type=float, default=2000.0, help='Style weight')
# parser.add_argument('--tv-weight', type=float, default=100000.0, help='Total variation regularization weight')
# NOTE: High tv-weight is bad and makes all images a solid color
# Not working, lr=0.005
# parser.add_argument('--content-weight', type=float, default=10, help='Content weight')
# parser.add_argument('--style-weight', type=float, default=2000.0, help='Style weight')
# parser.add_argument('--tv-weight', type=float, default=100000.0, help='Total variation regularization weight')
# Working 2, lr=0.001
# parser.add_argument('--content-weight', type=float, default=1, help='Content weight')
# parser.add_argument('--style-weight', type=float, default=200.0, help='Style weight')
# parser.add_argument('--tv-weight', type=float, default=200.0, help='Total variation regularization weight')

# For VGG19
parser.add_argument('--content-weight', type=float, default=15.0, help='Content weight')
# parser.add_argument('--style-weight', type=float, default=400.0, help='Style weight')
parser.add_argument('--style-weight', type=float, default=400.0, help='Style weight')
# parser.add_argument('--style-weight', type=float, default=500.0, help='Style weight')
# For Resnet18
# parser.add_argument('--content-weight', type=float, default=15.0*100, help='Content weight')
# parser.add_argument('--style-weight', type=float, default=100.0*100000000/4, help='Style weight')
# parser.add_argument('--style-weight', type=float, default=0.1, help='Style weight')
# Default weights
# parser.add_argument('--content-weight', type=float, default=15.0, help='Content weight')
# parser.add_argument('--style-weight', type=float, default=100.0, help='Style weight')
parser.add_argument('--tv-weight', type=float, default=200.0, help='Total variation regularization weight')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
# parser.add_argument('--lr', type=float, default=0.0015, help='Learning rate')
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

# style_layers = ['stage1_unit2_conv2', 'stage2_unit2_conv2', 'stage3_unit2_conv2', 'stage4_unit2_conv2']
# content_layer = 'stage4_unit1_conv2'
# content_layer = 'stage4_unit2_conv1'
# content_layer = 'stage4_unit2_conv2'
# style_layers = ['stage1_unit1_conv1', 'stage2_unit1_conv1', 'stage3_unit1_conv1', 'stage4_unit1_conv1'] #NOTE Use this
# style_layers = ['stage1_unit1_conv1', 'stage2_unit1_conv1']
# style_layers = ['stage3_unit1_conv1', 'stage4_unit1_conv1']
# style_layers = ['stage1_unit2_conv1', 'stage2_unit2_conv1', 'stage3_unit2_conv1', 'stage4_unit2_conv1']
# content_layer = 'stage4_unit2_conv2'

# style_layers = ['stage1_unit1_relu2', 'stage2_unit1_relu2', 'stage3_unit1_relu2', 'stage4_unit1_relu2']
# style_layers = ['stage1_unit1_relu2', 'stage1_unit2_relu2', 'stage2_unit1_relu2', 'stage2_unit2_relu2', 'stage3_unit1_relu2', 'stage3_unit2_relu2', 'stage4_unit1_relu2', 'stage4_unit2_relu2']
# style_layers = ['stage1_unit1_relu2', 'stage4_unit1_relu2']
# style_layers = ['add_1', 'add_2', 'add_3', 'add_4', 'add_5', 'add_6', 'add_7', 'add_8']
# # content_layer = 'stage3_unit2_relu2'
# # content_layer = 'stage4_unit2_relu2'
# # content_layer = 'stage4_unit1_relu2'
# content_layer = 'relu1'
# ResNet18, preprocess_input = Classifiers.get('resnet18')
# model = ResNet18((None, None, 3), weights='imagenet', include_top=False)
# model.summary()

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
# style_layers = ['block1_conv1', 'block3_conv1', 'block5_conv1']
content_layer = 'block4_conv2'
# style_layers = ['block1_conv1']
# content_layer = 'block2_conv2'
model = vgg19.VGG19(weights='imagenet', include_top=False, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)

# style_layers = ['block_4_expand_relu', 'block_8_expand_relu', 'block_12_expand_relu', 'block_16_expand_relu']
# content_layer = 'block_14_expand_relu'
# style_layers = ['block_2_expand_relu', 'block_4_expand_relu']
# content_layer = 'block_3_expand_relu'
# model = mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)

# style_layers = ['res2a_branch2a', 'res2b_branch2a', 'res3a_branch2a', 'res4a_branch2a', 'res5a_branch2a']
# style_layers = ['res2a_branch1', 'res3a_branch1', 'res4a_branch1', 'res5a_branch1']
# content_layer = 'res4f_branch2c'
# style_layers = ['res2a_branch1', 'res3a_branch1']
# content_layer = 'res3a_branch2a'
# model = resnet50.ResNet50(weights='imagenet', include_top=False, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
# Remove the last pooling layer
model = Model(model.input, model.layers[-2].output)
model.summary()

# x = Conv2D(32, 9, strides=(1, 1), padding='same', activation='relu')(model.input)
# x = Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2D(128, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2D(256, 3, strides=(2, 2), padding='same', activation='relu')(x)
# # x = Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu')(x)
# x = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2DTranspose(32, 3, strides=(2, 2), padding='same', activation='relu')(x)
# transform_output = Conv2D(3, 9, strides=(1, 1), padding='same', activation='sigmoid')(x)# * 150 + 255.0/2
# print(model.input, transform_output)
# transform_model = Model(model.input, transform_output)
# transform_model.summary()

x = Conv2D(32, 9, strides=(1, 1), padding='same', activation='relu')(model.input)
x = Conv2D(64, 7, strides=(2, 2), padding='same', activation='relu')(x)
x = Conv2D(128, 5, strides=(2, 2), padding='same', activation='relu')(x)
x = Conv2D(256, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu')(x)
x = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation='relu')(x)
x = Conv2DTranspose(64, 5, strides=(2, 2), padding='same', activation='relu')(x)
x = Conv2DTranspose(32, 7, strides=(2, 2), padding='same', activation='relu')(x)
transform_output = Conv2D(3, 9, strides=(1, 1), padding='same', activation='sigmoid')(x)# * 150 + 255.0/2
print(model.input, transform_output)
transform_model = Model(model.input, transform_output)
transform_model.summary()

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


# x = Conv2D(32, 9, strides=(1, 1), padding='same', activation='relu')(model.input)
# x = Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2D(128, 3, strides=(2, 2), padding='same', activation='relu')(x)
# # x = Conv2D(256, 3, strides=(2, 2), padding='same', activation='relu')(x)
#  # Residual blocks
# x = Add()([x, Conv2D(128, 3, strides=(1, 1), padding='same', activation='linear')(Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(x))])
# x = Add()([x, Conv2D(128, 3, strides=(1, 1), padding='same', activation='linear')(Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(x))])
# x = Add()([x, Conv2D(128, 3, strides=(1, 1), padding='same', activation='linear')(Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(x))])
# # x = Add()([x, Conv2D(128, 3, strides=(1, 1), padding='same', activation='linear')(Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(x))])
# # x = Add()([x, Conv2D(128, 3, strides=(1, 1), padding='same', activation='linear')(Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu')(x))])
# # x = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu')(x)
# x = Conv2DTranspose(32, 3, strides=(2, 2), padding='same', activation='relu')(x)
# transform_output = Conv2D(3, 9, strides=(1, 1), padding='same', activation='sigmoid')(x)# * 150 + 255.0/2
# print(model.input, transform_output)
# transform_model = Model(model.input, transform_output)


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

preds = transform_model(content_input / 255.0) * 255.0
test_preds = transform_model(test_input / 255.0) * 255.0
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
    # size = K.cast(height * width * filters, 'float32')
    # print(bs, height, width, filters, size)
    # feats = K.reshape(layer_output, (bs, height * width, filters))
    # feats_T = K.permute_dimensions(K.transpose(feats), [2, 0, 1])
    # print(feats, feats_T) # (20, 65536, 64) (20, 64, 65536)
    # grams = K.sum(K.dot(feats_T, feats) / size, axis=2)
    # grams = feats_T * feats / size
    # feats = K.reshape(layer_output, (bs * filters, width * height))
    # grams = K.dot(feats, K.transpose(feats)) / size
    size = K.cast(bs * height * width * filters, 'float32')
    feats = K.reshape(layer_output, (bs * width * height, filters))
    grams = K.dot(K.transpose(feats), feats) / size
    # grams = tf.matmul(feats_T, feats) / size
    # grams = K.concatenate([K.expand_dims(K.dot(feats_T[b, :, :], feats[b, :, :]), axis=0) for b in range(batch_size)], axis=0) / size
    # matmul = Lambda(lambda x: tf.matmul(x[0], x[1]))
    # grams = matmul([feats_T, feats]) / size
    style_gram = style_features[layer]
    # print(grams, style_gram.shape, style_gram.size) # (20, 64, 64) (64, 64) 4096
    # style_losses.append(l2_loss(grams - style_gram) / style_gram.size)
    # style_loss += l2_loss(grams - style_gram) / style_gram.size
    style_gram_size = K.cast(K.prod(K.shape(style_gram)), 'float32')
    style_loss += l2_loss(grams - style_gram) / style_gram_size
    # style_loss += l2_loss(grams - style_gram)
    # style_loss += l2_loss(grams - K.reshape(style_gram, ())) / style_gram_size
    # style_loss += l2_loss(grams) / style_gram.size
# print(style_losses)
# style_loss = args.style_weight * K.sum(style_losses) / batch_size
style_loss = args.style_weight * style_loss / batch_size

# # tv_y_size = K.cast(K.prod(K.shape(preds[:,1:,:,:])), 'float32')
# # tv_x_size = K.cast(K.prod(K.shape(preds[:,:,1:,:])), 'float32')
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

param_updates_content = adam.get_updates(params=transform_model.trainable_weights, loss=content_loss)
train_batch_content = K.function(inputs=[K.learning_phase(), content_input], outputs=[content_loss], updates=param_updates_content)

X_test = np.array([get_image(args.test)])
print(np.min(X_test), np.max(X_test))
print('Test image size:', np.shape(X_test))
print('Beginning!', len(content_targets) // batch_size, 'batches per epoch')
train_phase = 1
test_phase = 0
begin_time = time.time()

# TODO: First train transform model with only content loss to learn the identity mapping, then fine tune with the full loss function
for iteration in range(1000):
    start_time = time.time()
    curr = iteration * batch_size
    step = curr + batch_size
    X_batch = np.zeros(batch_shape, dtype=np.float32)
    for j, image in enumerate(content_targets[curr:step]):
        X_batch[j] = get_image(image, (256,256,3)).astype(np.float32)
    c_loss = train_batch_content([train_phase, X_batch])[0]
    if debug:
        print('Batch time: %.2f' % (time.time() - start_time), 'total time: %.2f' % (time.time() - begin_time), 'content:', c_loss)
    # TODO: Save checkpoint and test image every args.checkpoint_iterations
    if iteration % args.checkpoint_iterations == 0:
        t = time.time()
        transform_model.save(os.path.join(args.checkpoint_dir, os.path.splitext(os.path.basename(args.style))[0] + ('_content_%d.h5' % iteration)))
        test_image = test_batch([test_phase, X_test])[0][0]# * 255.0
        test_image = test_image.astype(int)
        save_image(os.path.join(args.test_dir, os.path.splitext(os.path.basename(args.test))[0] + ('_content_%d.jpg' % iteration)), test_image)
        print('Save test image time:', time.time() - t)


# TODO: !! Incrementally increase the weight for the style loss. This is like curriculum learning


for epoch in range(epochs):
    for iteration in range(len(content_targets) // batch_size):
        start_time = time.time()
        curr = iteration * batch_size
        step = curr + batch_size
        X_batch = np.zeros(batch_shape, dtype=np.float32)
        for j, image in enumerate(content_targets[curr:step]):
            X_batch[j] = get_image(image, (256,256,3)).astype(np.float32)
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


# style: 49218520.0, content:7287237.0, tv: 4244455.5
# content: 1825925.0 style: 3196250.0 tv: 8737.023
# content: 7634459.5 style: 33158612.0 tv: 18535.094
