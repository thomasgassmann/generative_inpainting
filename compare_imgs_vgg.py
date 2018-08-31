import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
from common_utils import get_files_in_dir
from PIL import Image
import tensorflow as tf
import tensorlayer as tl
from model_vgg import Vgg19_simple_api


################ Testing
# get all images in our train set
def get_all_images_in_dirs(dirs):
    img_paths = []
    for dir in dirs:
        img_paths.extend(get_files_in_dir(dir, True))
    return img_paths


img_files = get_all_images_in_dirs(dirs=[
    'training_data/training/celebsa',
    'training_data/validation/celebsa'
])
test_imgs = get_all_images_in_dirs(dirs=['test_data_use'])
all_imgs = test_imgs + img_files
filename_queue = tf.train.string_input_producer(all_imgs, shuffle=False)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
curr_img = tf.image.decode_png(value)
curr_img = tf.reshape(curr_img, [1, 64, 64, 3])
curr_img = tf.cast(curr_img, tf.float32)
curr_img = tf.Print(curr_img, [key, curr_img, "current processed image"])
## train inference
t_target_image = tf.placeholder('float32', [1, 64, 64, 3], name='t_target_image')
t_predict_image = tf.placeholder('float32', [1, 64, 64, 3], name='t_predict_image')


## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
t_target_image_224 = tf.image.resize_images(
    t_target_image,
    size=[224, 224],
    method=0,
    align_corners=False)  # resize_target_image_for_vgg
t_predict_image_224 = tf.image.resize_images(
    t_predict_image,
    size=[224, 224],
    method=0,
    align_corners=False)  # resize_generate_image_for_vgg

net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
_, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)
vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

top3_imgs_per_tensor = [[] for i in range(len(test_imgs))]
top3_loss_per_tensor = [[] for j in range(len(test_imgs))]

with tf.Session() as sess:
    tf.global_variables_initializer()

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)

    ###============================= COMPUTE ===============================###

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    t_img_tensors = []
    for i in range(len(all_imgs)):  # length of your filename list
        img = curr_img.eval()
        if i < len(test_imgs):
            t_img_tensors.append(img)
            print("adding processed tensor to test img tensors")
        else:
            print("comparing processed tensor")
            for t in range(len(t_img_tensors)):
                print("current closest 3 images for tesnor %d: %s" % (t, str(top3_imgs_per_tensor[t])))
                print("current closest 3 losses for tesnor %d: %s" % (t, str(top3_loss_per_tensor[t])))
                res = sess.run(vgg_loss, feed_dict={t_target_image: t_img_tensors[t], t_predict_image: img})
                print("got loss %s for test tensor %d" % (str(res), t))
                if len(top3_imgs_per_tensor[t]) < 3:
                    print("updating top 3 for tensor %d" % t)
                    top3_imgs_per_tensor[t].append(all_imgs[i])
                    top3_loss_per_tensor[t].append(res.item())
                else:
                    max_loss = max(top3_loss_per_tensor[t])
                    if max_loss > res.item():
                        print("updating top 3")
                        # found better loss
                        # find index of previous worst loss and remove
                        idx = top3_loss_per_tensor[t].index(max_loss)
                        del top3_loss_per_tensor[t][idx]
                        del top3_imgs_per_tensor[t][idx]
                        top3_loss_per_tensor[t].append(res.item())
                        top3_imgs_per_tensor[t].append(all_imgs[i])

print("done comparing, printing results: ")
print("test images are:")
print(test_imgs)
print("top 3 per test image:")
print(top3_imgs_per_tensor)
print(top3_loss_per_tensor)
