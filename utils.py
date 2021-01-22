import numpy as np, os, sys, glob# , scipy.misc
import cv2
# import imageio

def save_image(out_path, image):
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, image)
    # scipy.misc.imsave(out_path, img)
    # imageio.imwrite.imsave(out_path, img)

# def scale_image(style_path, style_scale):
#     scale = float(style_scale)
#     o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
#     # o0, o1, o2 = imageio.imread(style_path, pilmode='RGB').shape
#     scale = float(style_scale)
#     new_shape = (int(o0 * scale), int(o1 * scale), o2)
#     style_target = get_image(style_path, img_size=new_shape)
#     return style_target

def get_image(src, image_size=False):
    image = cv2.imread(src)
    # image = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
    # image = imageio.imread(src, pilmode='RGB')
    if not (len(image.shape) == 3 and image.shape[2] == 3):
        image = np.dstack((image, image, image))
    if image_size != False:
        image = cv2.resize(image, image_size[:2])
        # image = scipy.misc.imresize(image, image_size)
    return image

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def list_files(in_path, query='*'):
    # return glob.glob(os.path.join(in_path, query))
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend([f for f in filenames if 'DS_Store' not in f])
        break

    return files

