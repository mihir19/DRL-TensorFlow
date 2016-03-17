# DRL-TensorFlow

DRL (Deep Reinforcement Learning Agent) is an implementation of the deep reinforcement learning algorithm, described in [Playing Atari with Deep Reinforcement Learning](http://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
paper by [DeepMind](http://deepmind.com/).
The original implementation of the [DRL project](https://github.com/DSG-SoftServe/DRL) was in theano and was based 
on [TNNF](http://tnnf.readthedocs.org/en/latest/)-Tiny Neural Net Framework which used theano library for GPU computing.

The Tiny Neural Net Framework was revised to work on TensorFlow and hence this DRL project works on TensorFlow instead of theano.
DRL is currently tested on Ubuntu 14.04 on GPU.

Games: [Breakout](https://en.wikipedia.org/wiki/Breakout_(video_game)) [(video)](http://youtu.be/T58HkwX-OuI) and [Space Invaders](https://en.wikipedia.org/wiki/Space_Invaders)


## Installation
To install **DRL** on Ubuntu:
```
sudo apt-get install python-pil python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git libsdl1.2-dev libsdl-image1.2-dev libsdl-gfx1.2-dev python-matplotlib libyaml-dev
sudo pip install -U numpy
sudo pip install -U pillow==2.7.0
```
