# ColoriChainer

## What is this?

Convert black and white frames in a movie into colorized one. This will do over learning for only a few frames in a short (4-5 sec) movie which doesn't have heavy moving so that it can convert all frames with high quality.

## Environment

- GPU
- Python 3
- OpenCV 3.2.0
    - with Python 3 plugin
- Chainer 2.0.0b1

## Installation

```
git clone (URL here)
pip install -r requirements.txt
```

Check carefully whether Chainer _2.0.0b1_ is installed correctly.

## Usage

### Training

```
python train.py --train --data-dir=/data/hoge/sea --epoch=8000  
```

Data directory should have "train" and "val" directories which has frames for training and validation (test).

### Testing

```
python train.py --test --data-dir=/data/hoge/sea --model=./result/snapshot_iter_XXXX.npz
```

For other options, see `python train.py --help`.

## Tips

### Create each frame's image files from a movie file

```
ffmpeg -i foo.mp4 -qscale:v 2 -vf fps=24 %04d.jpg
```

### Create a video from images

```
ffmpeg -i ~/%d.jpg -c:v prores_aw -profile:v 3 -pix_fmt yuv422p10 -r 24/1.001 foo.mov
```

Names of output/input image files can be formatted string.

See details in https://trac.ffmpeg.org/wiki/Create%20a%20thumbnail%20image%20every%20X%20seconds%20of%20the%20video and https://trac.ffmpeg.org/wiki/Slideshow .