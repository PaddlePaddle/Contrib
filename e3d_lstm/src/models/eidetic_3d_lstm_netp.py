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

"""Builds an E3D RNN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.layers.rnn_cellp import Eidetic3DLSTMCell as eidetic_lstm
# import tensorflow as tf
import paddle.fluid as fluid

def rnn(images, real_input_flag, num_layers, num_hidden, configs):
    """Builds a RNN according to the config."""
    gen_images, lstm_layer, cell, hidden, c_history = [], [], [], [], []
    shape = images.shape
    batch_size = shape[0]
    # seq_length = shape[1]
    ims_width = shape[2]
    ims_height = shape[3]
    output_channels = shape[-1]
    # filter_size = configs.filter_size
    total_length = configs.total_length
    input_length = configs.input_length

    window_length = 2
    window_stride = 1

    for i in range(num_layers):
        if i == 0:
            num_hidden_in = output_channels
        else:
            num_hidden_in = num_hidden[i - 1]
        new_lstm = eidetic_lstm(
            name='e3d' + str(i),
            # according to my observation, input will be (batches, steps, w, h, c)
            input_shape=[ims_width, window_length, ims_height, num_hidden_in],
            output_channels=num_hidden[i],
            kernel_shape=[2, 5, 5])
        lstm_layer.append(new_lstm)
        zero_state = fluid.layers.zeros(
            [batch_size, window_length, ims_width, ims_height, num_hidden[i]], dtype='float32')
        cell.append(zero_state)
        hidden.append(zero_state)
        c_history.append(None)

    memory = zero_state

    generator_scope = fluid.Scope()
    # with tf.variable_scope('generator'):
    with fluid.scope_guard(generator_scope):
        input_list = []
        reuse = False
        for time_step in range(window_length - 1):
            input_list.append(
                fluid.layers.zeros([batch_size, ims_width, ims_height, output_channels], dtype='float32'))

        # outer layer
        for time_step in range(total_length - 1):
            e3dlstm_scope = fluid.Scope()
            # with tf.variable_scope('e3d-lstm', reuse=reuse):
            with fluid.scope_guard(e3dlstm_scope):
                # all batches at time_step
                if time_step < input_length:
                    input_frm = images[:, time_step]
                else:
                    time_diff = time_step - input_length
                    input_frm = real_input_flag[:, time_diff] * images[:, time_step] \
                                + (1 - real_input_flag[:, time_diff]) * x_gen  # pylint: disable=used-before-assignment
                input_list.append(input_frm)

                # input_list becomes (steps, batches, w, h, c)
                if time_step % (window_length - window_stride) == 0:
                    # [time_step:] or [:time_step] ?
                    # After thinking a while, it seems correct: since at above code, zeros are added for
                    # padding, so current time_step is actually beind by (window_length - 1), then
                    # following input_frm is two step: last-1, last
                    input_frm = fluid.layers.stack(input_list[time_step:])
                    input_frm.stop_gradient = True
                    # (steps, batches, w, h, c) -> (batches, steps, w, h, c)
                    input_frm = fluid.layers.transpose(input_frm, [1, 0, 2, 3, 4])

                    # stacked RNN? it seems it's not stacked structure
                    # inner layer
                    for i in range(num_layers):
                        # for time step 0, cell[i] is all zeros
                        if time_step == 0:
                            c_history[i] = cell[i]
                        else:
                            # [c_history[i], cell[i]] or [c_history[i-1], cell[i-1]] ?
                            # above thoughts is likely incorrect
                            # it will load memories from previous step?
                            # actually it's correct
                            # after concatenating, the shape become
                            # [batch_size, window_length*2, ims_width, ims_height, num_hidden[i]]

                            # just found one thing, the c_history timestep dimension will increase at each
                            # timestamp because previous timestamp c_history is not reset (or cleared)
                            c_history[i] = fluid.layers.concat([c_history[i], cell[i]], 1)
                        if i == 0:
                            inputs = input_frm
                        else:
                            inputs = hidden[i - 1]

                        # so hidden[i], cell[i], memory, c_history[i] is all global scope,
                        # hidden[i], cell[i], c_history[i] is updated from previous time step,
                        # only inputs is loaded from previous layer
                        hidden[i], cell[i], memory = lstm_layer[i](
                            inputs, hidden[i], cell[i], memory, c_history[i])

                x_gen = fluid.layers.conv3d(hidden[num_layers - 1], output_channels,
                                            (window_length, 1, 1), (window_length, 1, 1),
                                         'same', data_format='NDHWC')
                print(x_gen.shape)
                x_gen = fluid.layers.squeeze(x_gen, [])
                print(x_gen.shape)
                gen_images.append(x_gen)
                reuse = True

    # (timesteps, batches, w, h, c)
    gen_images = fluid.layers.stack(gen_images)
    # (timesteps, batches, w, h, c) -> (batches, timesteps, w, h, c)
    gen_images = fluid.layers.transpose(gen_images, [1, 0, 2, 3, 4])
    # loss = tf.nn.l2_loss(gen_images - images[:, 1:])
    # loss += tf.reduce_sum(tf.abs(gen_images - images[:, 1:]))
    # TODO: the original loss seems to use sum of l2 loss and l1 loss, my loss is not exactly same
    # loss = fluid.layers.mse_loss(gen_images, images[:, 1:])
    # use the loss from original code
    loss = fluid.layers.reduce_sum(fluid.layers.square_error_cost(gen_images, images[:, 1:])) / 2
    loss += fluid.layers.reduce_sum(fluid.layers.abs(gen_images - images[:, 1:]))
    loss = loss / batch_size
    loss = loss / 2**7

    out_len = total_length - input_length
    out_ims = gen_images[:, -out_len:]

    # fluid.layers.Print(loss, message='loss: ')

    return [out_ims, loss]
