# -*- coding: UTF-8 -*-

import os.path
import time
import sys
from optparse import OptionParser

import chainer
import cupy as cp
import cv2
import numpy as np
from chainer import configuration
from chainer import training
from chainer.training import extensions
import skimage.io

from dataset import Pix2PixDataset
from net_cycle import *
from updater import DCGANUpdater

import save_parameter 


def test(data_dir: str, gen, compare: bool, fps: int, use_original: bool):
    def array_to_image(arr):
        return np.asarray((chainer.cuda.to_cpu(cp.transpose((cp.clip(arr, -1, 1) + 1) * 127.5, (1, 2, 0)))),
                          dtype=np.uint8)

    width = 540
    test_imgs = Pix2PixDataset(os.path.join(data_dir, 'val'), width, train=False, use_original=use_original)
    data_dir_name = os.path.split(data_dir)[1]
    #video_writer = cv2.VideoWriter('./test/' + data_dir_name + '.mp4', cv2.VideoWriter_fourcc(*'H264'),
#                               fps, (1920 * 2, 1080), False)
    l = len(test_imgs)
    start = time.time()
    for img_i in range(l):
        bw, real = test_imgs.get_example(img_i)
        test_img = cp.asarray([bw])
        output = gen(test_img, False).data[0]
        if compare:
            bw_img = np.concatenate([array_to_image(bw)] * 3, axis=2)
            fake_img = array_to_image(output)
            real_img = array_to_image(real)
            output_img = np.concatenate([bw_img, fake_img, real_img], axis=1)
        else:
            output_img = array_to_image(output)

        skimage.io.imsave('./test/{}.tif'.format(img_i), output_img)
#        video_writer.write(cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        print(img_i + 1, '/', l)

    end = time.time()
    duration = end - start
    print(duration, 'sec')
    print(l / duration, 'fps')


def main():
    parser = OptionParser()
    parser.add_option('--data-dir', help='directory including train/val directries', type='string')
    parser.add_option('--train', help='train mode', action='store_true', dest='is_train', default=True)
    parser.add_option('--test', help='test mode', action='store_false', dest='is_train')
    parser.add_option('--model', help='npz file of trained model', type='string')
    parser.add_option('--compare', help='compare fake and real in test mode', action='store_true',
                      default=False)
    parser.add_option('--epoch', help='number of epoch', default=1000, type='int')
    parser.add_option('--fps', help='fps of input data', default=24, type='int')
    parser.add_option('--trigger', help='number of epoch where model is saved', default=100, type='int')
    #parser.add_option('--user-model', help='directory where user-created models are stored', default='',type='str')
    parser.add_option('--use-original', help='use original BW images', action='store_true', default=False)

    options, args = parser.parse_args()

    configuration.config.train = options.is_train

    train = Pix2PixDataset(os.path.join(options.data_dir, 'train'), 540, train=False, use_original=options.use_original)
    train_iter = chainer.iterators.SerialIterator(train, 1)
    gen = Generator()
    dis = Discriminator()
    gen.to_gpu()
    dis.to_gpu()

    for i in range(10,options.epoch + 1,10):
        num_test = i * save_parameter.load_example_number.number_of_frames
        for s in range(save_parameter.load_example_number.number_of_frames):
            save_parameter.load_example_number.img_save_iteration.append(num_test-s)
            #save_parameter.load_example_number.img_save_iteration.append(num_test-1)
            #save_parameter.load_example_number.img_save_iteration.append(num_test)

    def make_optimizer(model, alpha=0.0001, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.use_cleargrads()
        # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer

    def make_optimizer_gen(model, alpha=0.0001, beta1=0.9):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.use_cleargrads()
        #optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer

    def make_optimizer_dis(model, lr=0.001):
        optimizer = chainer.optimizers.SGD(lr=lr)
        optimizer.setup(model)
        optimizer.use_cleargrads()
        #optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer_gen(gen)
    opt_dis = make_optimizer_dis(dis)

    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=0)
    trainer = training.Trainer(updater, (options.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
#    trainer.extend(extensions.PrintReport(
#        ['epoch', 'iteration', 'gen/loss', 'gen/distance', 'dis/loss', 'dis/real_loss', 'dis/fake_loss']),
#                   trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.epoch}.npz'), trigger=(options.trigger, 'epoch'))
    trainer.extend(extensions.PlotReport(['gen/loss', 'dis/loss'], marker=None, trigger=(10, 'epoch')))
    #trainer.extend(extensions.ProgressBar(update_interval=1))
    # trainer.extend(out_generated_image(gen, Pix2PixDataset('./C0149/val/', 240, train=False)), trigger=(1, 'epoch'))
    # trainer.extend(extensions.dump_graph('dis/loss'), trigger=(1, 'epoch'))

    if options.model:
        chainer.serializers.load_npz(options.model, trainer)
#        options.epoch += self.epoch_now
        print('[LOAD WEIGHT COMPLETED]')

    print('[TRAIN START]')
    if options.is_train:
        start = time.time()
        trainer.run()
    else:
        start = time.time()
        test(options.data_dir, trainer.updater.gen, options.compare, options.fps, use_original=options.use_original)

    end = time.time()

    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    if seconds > 0 and minutes == 0 and hours ==0:
        print("Total Elapsed Time: {:0>2} sec".format(int(seconds)))
    else:
        if minutes > 0 and hours == 0:
            print("Total Elapsed Time: {:0>2}m {:0>2}s".format(int(minutes),int(seconds)))
        else:
            print("Total Elapsed Time: {:0>2}h {:0>2}m {:0>2}s".format(int(hours),int(minutes),int(seconds)))


if __name__ == '__main__':
    main()
