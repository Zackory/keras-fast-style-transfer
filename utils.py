import numpy as np, os, sys, glob, scipy.misc
# import imageio

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)
    # imageio.imwrite.imsave(out_path, img)

def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    # o0, o1, o2 = imageio.imread(style_path, pilmode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target

def get_img(src, img_size=False):
   img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
   # img = imageio.imread(src, pilmode='RGB')
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img

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

