# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from utils.mr import masking
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]
        if self.args.concat == True:
            self.opt.zero_grad()
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform)
                inputs = torch.cat((inputs, buf_inputs))
                labels = torch.cat((labels, buf_labels))

            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels[:real_batch_size])
        else : 
            self.opt.zero_grad()
            if self.args.current_b == True:
                outputs =self.net.forward_adv(inputs)
            else:
                outputs = self.net(inputs)
            if self.args.masking_c ==True:
                outputs = masking(outputs,labels,self.device)
            loss = self.loss(outputs,labels)
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform)
                if self.args.memory_b == True:
                    b_outputs = self.net.forward_adv(buf_inputs)
                else:
                    b_outputs =self.net(buf_inputs)
                if self.args.masking_m == True:
                    b_outputs = masking(b_outputs,buf_labels,self.device)
                loss += self.loss(b_outputs,buf_labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels[:real_batch_size])            

        return loss.item()
