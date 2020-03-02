# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Factory to get E3D-LSTM models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from src.models import eidetic_3d_lstm_netp
import paddle.fluid as fluid
import numpy as np


class Model(object):
  """Model class for E3D-LSTM model."""

  def __init__(self, configs):
    self.configs = configs

    place = fluid.CUDAPlace(0)
    # place = fluid.CPUPlace()
    self.sess = fluid.Executor(place)
    self.start_program = fluid.Program()
    self.train_program = fluid.Program()

    with fluid.program_guard(self.train_program, self.start_program):

      self.x = fluid.data('imgs', [
              self.configs.batch_size, self.configs.total_length,
              self.configs.img_width // self.configs.patch_size,
              self.configs.img_width // self.configs.patch_size,
              self.configs.patch_size * self.configs.patch_size *
              self.configs.img_channel
          ])


      self.real_input_flag = fluid.data('real_input_flag', [
          self.configs.batch_size,
          self.configs.total_length - self.configs.input_length - 1,
          self.configs.img_width // self.configs.patch_size,
          self.configs.img_width // self.configs.patch_size,
          self.configs.patch_size * self.configs.patch_size *
          self.configs.img_channel
      ])

      self.pred_seq = []
      self.params = dict()
      num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
      num_layers = len(num_hidden)

      with fluid.unique_name.guard():
        output_list = self.construct_model(self.x, self.real_input_flag,
                                           num_layers, num_hidden)
        self.gen_ims = output_list[0]
        self.loss = output_list[1]

        self.pred_seq.append(self.gen_ims)

        self.test_program = self.train_program.clone(for_test=True)

        # adam = fluid.optimizer.Adam(self.configs.lr, regularization=fluid.regularizer.L2Decay(0.0005), beta1=0.95, beta2=0.9995)
        # changing optimizer from SGD to Adam for exsiting model failed, because Adam added parameters
        # to model: momentums
        adam = fluid.optimizer.SGD(self.configs.lr, regularization=fluid.regularizer.L2Decay(0.0005))
        self.train_op = adam.minimize(self.loss)

    self.sess.run(self.start_program)

    if self.configs.pretrained_model:
      fluid.io.load_persistables(self.sess, self.configs.pretrained_model, self.train_program)

  def train(self, inputs, lr, real_input_flag, itr):
    print(np.array(inputs).shape, np.array(real_input_flag).shape)
    inputs = np.squeeze(inputs)
    feed_dict = {'imgs': inputs}
    feed_dict.update({'real_input_flag': real_input_flag})
    # self.sess.run(fluid.default_startup_program())
    print('start training')
    _, loss = self.sess.run(self.train_program, feed=feed_dict, fetch_list=[self.gen_ims, self.loss])
    # print('result: ', loss, _)
    print('after training', )
    return loss

  def test(self, inputs, real_input_flag):
    feed_dict = {'imgs': inputs}
    feed_dict.update({'real_input_flag': real_input_flag})
    gen_ims = self.sess.run(self.test_program, feed=feed_dict, fetch_list=[self.gen_ims])
    gen_ims = gen_ims
    return gen_ims

  def save(self, itr):
    save_path = os.path.join(self.configs.save_dir, str(itr))
    fluid.io.save_persistables(self.sess, save_path, self.train_program)
    print('saved to ' + self.configs.save_dir)

  def load(self, checkpoint_path):
    print('load model:', checkpoint_path)
    fluid.io.load_persistables(self.sess, checkpoint_path, self.train_program)

  def construct_model(self, images, real_input_flag, num_layers, num_hidden):
    """Contructs a model."""
    networks_map = {
        'e3d_lstm': eidetic_3d_lstm_netp.rnn,
    }

    if self.configs.model_name in networks_map:
      func = networks_map[self.configs.model_name]
      return func(images, real_input_flag, num_layers, num_hidden, self.configs)
    else:
      raise ValueError('Name of network unknown %s' % self.configs.model_name)
