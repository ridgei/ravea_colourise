import chainer
import chainer.functions as F
import cupy as cp
import numpy as np
import skimage.color
import skimage.io
import os
from PIL import Image
from chainer import Variable
from ganhacks import noise
import save_parameter

class DCGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = y_fake.data.shape[0]
        # L1 = F.sum(F.softplus(-y_real)) / batchsize
        L1 = F.mean_squared_error(y_real, noise.noisy_label(batchsize, mode='real').reshape(batchsize, 1))
        # L2 = F.sum(F.softplus(y_fake)) / batchsize
        L2 = F.mean_squared_error(y_fake, noise.noisy_label(batchsize, mode='fake').reshape(batchsize, 1))
        loss = L1 + L2
        chainer.report({'loss': loss, 'real_loss': L1, 'fake_loss': L2}, dis)
        return loss

    def loss_gen(self, gen, y_fake, x_real, x_fake):
        batchsize = y_fake.data.shape[0]
        # loss = F.sum(F.softplus(-y_fake)) / batchsize
        fake_loss = F.mean_squared_error(y_fake, cp.ones((batchsize, 1), cp.float32))
        distance = F.mean_absolute_error(x_real, x_fake)
        loss = fake_loss + distance
        chainer.report({'loss': loss, 'fake_loss': fake_loss, 'distance': distance}, gen)
        return loss

    def array_to_image(self, arr):
        return np.asarray((chainer.cuda.to_cpu(cp.transpose((cp.clip(arr, -1, 1) + 1) * 127.5, (1, 2, 0)))),
                          dtype=np.uint8)

    def update_core(self):
        iteration = self.iteration+1
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        batch = self.get_iterator('main').next()
        gen, dis = self.gen, self.dis
        z, x = self.converter(batch, self.device)
        # xp = chainer.cuda.get_array_module(x_real.data)
        # z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_real = Variable(x)
        x_fake = gen(Variable(z), train=True)
        #print("[EXAMPLE_NO]:",save_parameter.load_example_number.example_no)
        epoch = self.epoch
        no_frames = save_parameter.load_example_number.number_of_frames
        if iteration % no_frames == 0:
            print("[EPOCH]:", self.epoch)
        else:
            print("[EPOCH]:", self.epoch + 1)
        print("[ITERATION]:", iteration)
        print("[FRAMES_NO]:", no_frames)
#        print("[SAVE_ITERATION]:",save_parameter.load_example_number.img_save_iteration)
        if iteration in save_parameter.load_example_number.img_save_iteration:
#            z_image = np.concatenate([self.array_to_image(z[0])] * 1, axis=2)
#            x_real_image = self.array_to_image(x[0])
#            x_fake_image = self.array_to_image(x_fake.data[0])
#            concatenated_images = np.concatenate([z_image, x_fake_image, x_real_image], axis=1)
#            skimage.io.imsave('./sample/{}.jpg'.format(self.epoch), concatenated_images)
            z_image = np.concatenate([self.array_to_image(z[0])] * 1, axis=2)
            x_real_image = self.array_to_image(x[0])
            x_fake_image = self.array_to_image(x_fake.data[0])
            #print(z.shape, x.shape, )
            concatenated_images = np.concatenate([z_image, x_fake_image, x_real_image], axis=1)
            if self.epoch % 10 == 9:
                skimage.io.imsave('./sample/{}_{:0>1}.jpg'.format(self.epoch+1, save_parameter.load_example_number.example_no), concatenated_images)
            else:
                skimage.io.imsave('./sample/{}_{:0>1}.jpg'.format(self.epoch, save_parameter.load_example_number.example_no), concatenated_images)
            skimage.io.imsave('./sample/current.jpg'.format(self.epoch), concatenated_images)

        y_real = dis(Variable(cp.concatenate((z, x), axis=1)), train=True)
        y_fake = dis(Variable(cp.concatenate((z, x_fake.data), axis=1)), train=True)

        gen_optimizer.update(self.loss_gen, gen, y_fake, x_real, x_fake)
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
