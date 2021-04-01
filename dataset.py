import glob
import os

import chainer
import numpy
import scipy.misc
import skimage.color
import skimage.io
import skimage.util
import save_parameter


class Pix2PixDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, crop_size, train=True, use_original=False):
        self._paths = list(sorted(glob.glob(os.path.join(path, '*.tif'))))
        self._crop_size = crop_size
        self._train = train
        self._use_original = use_original
        save_parameter.load_example_number.number_of_frames = len(self._paths)
        #save_parameter.load_example_number.img_save_iteration = [x for x in range(len(self._paths))]

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        print("----------------------------------------")
        save_parameter.load_example_number.example_no = i
        image_size = (1080, 1920)
        image_path = self._paths[i]
        image = skimage.io.imread(image_path)
        image = scipy.misc.imresize(image, image_size)

        if self._train:
            h, w, c = image.shape
            crop_x = numpy.random.randint(h - self._crop_size)
            crop_y = numpy.random.randint(w - self._crop_size)
            image = image[crop_x:crop_x+self._crop_size,
                                  crop_y:crop_y+self._crop_size]
            if numpy.random.random() < 0.5:
                image = numpy.fliplr(image)

        if self._use_original:
            color_image_dir, color_image_fn = os.path.split(image_path)
            orig_image_file = os.path.join(color_image_dir, 'orig', color_image_fn)
            orig_image = skimage.io.imread(orig_image_file, as_grey=False)
            orig_image = scipy.misc.imresize(orig_image, image_size)
            return self._get_orig_color_pair(orig_image, image)
        else:
            return self._get_gray_color_pair(image)

    def get_random_example(self):
        i = numpy.random.randint(self.__len__())
        return self.get_example(i)

    def _get_gray_color_pair(self, img_arr):
        # color = numpy.asarray(skimage.color.rgb2hsv(img_arr), dtype=numpy.float32)
        color = numpy.asarray(img_arr, dtype=numpy.float32) / 127.5 - 1
        # gray = color[:, :, 2, None]
        #gray = skimage.color.rgb2gray(color)
        # gray = skimage.util.random_noise(gray)
        #gray = numpy.asarray(gray[:, :, None], dtype=numpy.float32)
        
        return color.transpose(2, 0, 1), color.transpose(2, 0, 1)

    def _get_orig_color_pair(self, orig_arr, color_arr):
        orig = numpy.asarray(orig_arr, dtype=numpy.float32) / 127.5 - 1
        color = numpy.asarray(color_arr, dtype=numpy.float32) / 127.5 - 1
        return (orig.transpose(2, 0, 1), color.transpose(2, 0, 1))
