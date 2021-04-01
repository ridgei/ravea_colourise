import chainer
import chainer.links as L
import chainer.functions as F

class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            c1=L.Convolution2D(None, 64, 4, 2, 1),
            c2=L.Convolution2D(None, 128, 4, 2, 1),
            c3=L.Convolution2D(None, 256, 4, 2, 1),
            c4=L.Convolution2D(None, 512, 4, 2, 1),
            c5=L.Convolution2D(None, 512, 4, 2, 1),
            c6=L.Convolution2D(None, 512, 4, 2, 1),
            c7=L.Convolution2D(None, 512, 4, 2, 1),
            dc6=L.Deconvolution2D(None, 512, 4, 2, 1),
            dc5=L.Deconvolution2D(None, 512, (3, 4), 2, (0, 1)),
            dc4=L.Deconvolution2D(None, 512, (3, 4), 2, (0, 1)),
            dc3=L.Deconvolution2D(None, 256, (3, 4), 2, (0, 1)),
            dc2=L.Deconvolution2D(None, 128, 4, 2, 1),
            dc1=L.Deconvolution2D(None, 64, 4, 2, 1),
            dc0=L.Deconvolution2D(None, 3, 4, 2, 1),
            bnc1=L.BatchNormalization(64),
            bnc2=L.BatchNormalization(128),
            bnc3=L.BatchNormalization(256),
            bnc4=L.BatchNormalization(512),
            bnc5=L.BatchNormalization(512),
            bnc6=L.BatchNormalization(512),
            bnc7=L.BatchNormalization(512),
            bndc6=L.BatchNormalization(512),
            bndc5=L.BatchNormalization(512),
            bndc4=L.BatchNormalization(512),
            bndc3=L.BatchNormalization(256),
            bndc2=L.BatchNormalization(128),
            bndc1=L.BatchNormalization(64)
        )

    def __call__(self, pix, train):
        conv1 = F.leaky_relu(self.bnc1(self.c1(pix)))
        conv2 = F.leaky_relu(self.bnc2(self.c2(conv1)))
        conv3 = F.leaky_relu(self.bnc3(self.c3(conv2)))
        conv4 = F.leaky_relu(self.bnc4(self.c4(conv3)))
        conv5 = F.leaky_relu(self.bnc5(self.c5(conv4)))
        conv6 = F.leaky_relu(self.bnc6(self.c6(conv5)))
        conv7 = self.bnc7(self.c7(conv6))
        deconv6 = F.dropout(F.leaky_relu(self.bndc6(self.dc6(conv7))))
        deconv6 = F.concat([conv6, deconv6])
        deconv5 = F.dropout(F.leaky_relu(self.bndc5(self.dc5(deconv6))))
        deconv5 = F.concat([conv5, deconv5])
        deconv4 = F.dropout(F.leaky_relu(self.bndc4(self.dc4(deconv5))))
        deconv4 = F.concat([conv4, deconv4])
        deconv3 = F.leaky_relu(self.bndc3(self.dc3(deconv4)))
        deconv3 = F.concat([conv3, deconv3])
        deconv2 = F.leaky_relu(self.bndc2(self.dc2(deconv3)))
        deconv2 = F.concat([conv2, deconv2])
        deconv1 = F.leaky_relu(self.bndc1(self.dc1(deconv2)))
        deconv1 = F.concat([conv1, deconv1])
        deconv0 = self.dc0(deconv1)
        output = F.tanh(deconv0)

        return output


class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c1=L.Convolution2D(None, 64, 4, 2, 1),
            c2=L.Convolution2D(None, 128, 4, 2, 1),
            c3=L.Convolution2D(None, 256, 4, 2, 1),
            c4=L.Convolution2D(None, 512, 4, 2, 1),
            c5=L.Convolution2D(None, 512, 4, 2, 1),
            c6=L.Convolution2D(None, 512, 4, 2, 1),
            c7=L.Convolution2D(None, 512, 4, 2, 1),
            bc1=L.BatchNormalization(64),
            bc2=L.BatchNormalization(128),
            bc3=L.BatchNormalization(256),
            bc4=L.BatchNormalization(512),
            bc5=L.BatchNormalization(512),
            bc6=L.BatchNormalization(512),
            bc7=L.BatchNormalization(512),
            l=L.Linear(None, 1)
        )

    def __call__(self, pix, train):
        h = F.leaky_relu(self.bc1(self.c1(pix)))
        h = F.leaky_relu(self.bc2(self.c2(h)))
        h = F.leaky_relu(self.bc3(self.c3(h)))
        h = F.leaky_relu(self.bc4(self.c4(h)))
        h = F.leaky_relu(self.bc5(self.c5(h)))
        h = F.leaky_relu(self.bc6(self.c6(h)))
        h = F.leaky_relu(self.bc7(self.c7(h)))
        h = self.l(h)
        return h
