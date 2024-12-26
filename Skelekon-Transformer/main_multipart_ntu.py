#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
from model.baseline import FinalModel

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

#from torchlight import DictAction
#import DictAction
from tools import *
from prompt import *
from Loss import Loss
from mmcv.cnn.utils import flops_counter



class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')  #pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)
        

classes, num_text_aug, text_dict = text_prompt_openai_pasta_pool_4part()
text_list = text_prompt_openai_random()



device = "cuda" if torch.cuda.is_available() else "cpu"

scaler = torch.cuda.amp.GradScaler()

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=80,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=50,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--loss-alpha', type=float, default=0.8)
    parser.add_argument('--te-lr-ratio', type=float, default=1)




    

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

        




        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                for name in self.arg.model_args['head']:
                    self.model_text_dict[name] = nn.DataParallel(
                        self.model_text_dict[name],
                        device_ids=self.arg.device,
                        output_device=self.output_device)
        


    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        # 获取输出设备
        output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
        self.output_device = output_device

        # 动态导入模型类
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)  # 保存模型代码文件到工作目录

        # 初始化模型
        self.model = Model(**self.arg.model_args)

        # 假设输入的形状是图像序列，修改输入形状为 (B, C, T, V, M)，具体根据你的需求来
        input_shape = (self.arg.batch_size, 3, 64, 25, 2)  # (batch_size, channels, time_steps, keypoints, persons)
        print(flops_counter.get_model_complexity_info(Model(**self.arg.model_args), input_shape))

        # 损失函数
        self.loss_ce = nn.CrossEntropyLoss().cuda(output_device)
        self.loss = Loss().cuda(output_device)

        # 模型
        self.model_text_dict = nn.ModuleDict()
        for name in self.arg.model_args['head']:
            model_, preprocess = device.load(name, self.arg.device)
            del model_.visual
            model_text = FinalModel(model_)
            model_text = model_text.cuda(self.output_device)
            self.model_text_dict[name] = model_text

        # 加载预训练模型权重（如果指定了权重文件）
        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])  # 提取全局步骤
            self.print_log(f'Load weights from {self.arg.weights}.')

            # 根据权重文件类型加载权重
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            # 将权重移动到GPU
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            # 处理需要忽略的权重
            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log(f'Successfully removed weights: {key}.')
                        else:
                            self.print_log(f'Cannot remove weights: {key}.')

            # 更新模型的状态字典并加载权重
            try:
                self.model.load_state_dict(weights)
            except Exception as e:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Cannot find these weights:')
                for d in diff:
                    print(f'  {d}')
                state.update(weights)
                self.model.load_state_dict(state)

        for name, model_text in self.model_text_dict.items():
            if f"{name}_weights" in self.arg:
                model_weights = self.arg[f"{name}_weights"]
                if os.path.exists(model_weights):
                    model_text.model.load_state_dict(torch.load(model_weights))
                    self.print_log(f"Loaded {name} weights from {model_weights}.")

            

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                [{'params': self.model.parameters(),'lr': self.arg.base_lr},
                {'params': self.model_text_dict.parameters(), 'lr': self.arg.base_lr*self.arg.te_lr_ratio}],
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        for i, data in enumerate(self.data_loader['train']):
            # Get data from the dataloader
            inputs, labels = data
            inputs = inputs.to(self.arg.device)
            labels = labels.to(self.arg.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.compute_loss(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1

            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    f'Epoch [{epoch + 1}/{self.arg.num_epoch}], Step [{self.global_step}], Loss: {loss.item()}')

        # Save model if needed
        if save_model:
            self.save_model(epoch)

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        self.model.eval()
        correct = 0
        total = 0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for name in loader_name:
                for data in self.data_loader[name]:
                    inputs, labels = data
                    inputs = inputs.to(self.arg.device)
                    labels = labels.to(self.arg.device)

                    # Forward pass
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_outputs.append(outputs.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

        accuracy = correct / total
        self.print_log(f'Epoch [{epoch + 1}], Accuracy: {accuracy * 100:.2f}%')

        if save_score:
            self.save_results(all_outputs, all_labels, wrong_file, result_file)

    def compute_loss(self, outputs, labels):
        # Define your loss function, for example CrossEntropyLoss
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs, labels)

    def save_model(self, epoch):
        save_path = os.path.join(self.arg.work_dir, f'epoch_{epoch + 1}.pt')
        torch.save(self.model.state_dict(), save_path)
        self.print_log(f'Model saved to {save_path}')

    def load_best_model(self):
        # Load the best model
        weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-' + str(self.best_acc_epoch) + '*'))[0]
        weights = torch.load(weights_path)

        if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
            weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])

        self.model.load_state_dict(weights)

        wf = weights_path.replace('.pt', '_wrong.txt')
        rf = weights_path.replace('.pt', '_right.txt')
        self.arg.print_log = False
        self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
        self.arg.print_log = True

    def print_model_statistics(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print_log(f'Best accuracy: {self.best_acc}')
        self.print_log(f'Epoch number: {self.best_acc_epoch}')
        self.print_log(f'Model name: {self.arg.work_dir}')
        self.print_log(f'Model total number of params: {num_params}')
        self.print_log(f'Weight decay: {self.arg.weight_decay}')
        self.print_log(f'Base LR: {self.arg.base_lr}')
        self.print_log(f'Batch Size: {self.arg.batch_size}')
        self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
        self.print_log(f'seed: {self.arg.seed}')

    def save_results(self, outputs, labels, wrong_file, result_file):
        # Save the results of evaluation
        with open(result_file, 'w') as rf:
            for output, label in zip(outputs, labels):
                rf.write(f'Predicted: {output}, Ground Truth: {label}\n')

        # Save wrong predictions
        wrong_predictions = [(output, label) for output, label in zip(outputs, labels) if output != label]
        with open(wrong_file, 'w') as wf:
            for wp in wrong_predictions:
                wf.write(f'Predicted: {wp[0]}, Ground Truth: {wp[1]}\n')


    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Parameters: {count_parameters(self.model)}')

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)) and (
                            epoch + 1) > self.arg.save_epoch

                # Train the model
                self.train(epoch, save_model=save_model)

                # Evaluate the model
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # Load and evaluate the best model
            self.load_best_model()

            # Output model statistics
            self.print_model_statistics()

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))

            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
