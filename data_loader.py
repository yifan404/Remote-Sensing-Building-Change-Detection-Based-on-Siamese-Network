from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import sys
import math
import torch


def cut_image_into_poe_neg_pieces(sr1, sr2, label, mean, std, stride=None, width_size=None, height_size=None):
    """
    # Cut the picture into small pieces and divide into two categories
    :param sr1: sr1
    :param sr2: sr2
    :param label: label
    :param stride: width and height stride, it can not be zero.
    :param width_size: expected image size in the width.
    :param height_size: expected image size in the height
    :param mean: mean of tree channles of an image
    :param std: std of three channles of an image
    :return: sr1_list, sr2_list, label_list.
    """
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)     # convert color label to gray
    sr1 = normalize(sr1, mean, std)
    sr2 = normalize(sr2, mean, std)
    label = label.astype('float') / 255
    if stride is None or stride == 0:
        # stride can not be None!
        print('Stride can not be None or zero!')
        sys.exit(-1)
    if width_size is None or height_size is None:
        # width or height size can not be None!
        print('width or height size can not be None!')
        sys.exit(-1)
    h, w, c = sr1.shape         # get the shape
    height_steps = math.ceil((h - height_size) / stride + 1)
    wide_steps = math.ceil((w - width_size) / stride + 1)
    if wide_steps is 0 or height_steps is 0:
        print('Error, this is because stride equals 1 and image size is one larger than output size.')
        sys.exit(-1)
    if c == 3:
        height_fill = (height_steps - 1) * stride + height_size - h     # The number of pixels to fill in the height
        wide_fill = (wide_steps - 1) * stride + width_size - w           # The number of pixels to fill in the width
        # fill the border
        sr1 = cv2.copyMakeBorder(sr1, 0, height_fill, 0, wide_fill, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        sr2 = cv2.copyMakeBorder(sr2, 0, height_fill, 0, wide_fill, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        label = cv2.copyMakeBorder(label, 0, height_fill, 0, wide_fill, cv2.BORDER_CONSTANT, value=[0])
        np_sr1_pos = []                 # save the result of cut
        np_sr1_neg = []
        np_sr2_pos = []
        np_sr2_neg = []
        np_label_pos = []
        np_label_neg = []
        for i in range(height_steps):
            for j in range(wide_steps):
                label_change = label[i * stride:i * stride + height_size, j * stride:j * stride + width_size]
                sr1_pieces = sr1[i * stride:i * stride + height_size, j * stride:j * stride + width_size, :]
                sr2_pieces = sr2[i * stride:i * stride + height_size, j * stride:j * stride + width_size, :]
                # if the label is not change, just save.
                if np.all(label_change == 0):
                    np_sr1_neg.append(sr1_pieces.transpose(2, 0, 1))
                    np_sr2_neg.append(sr2_pieces.transpose(2, 0, 1))
                    np_label_neg.append(label_change)
                else:
                    np_sr1_pos.append(sr1_pieces.transpose(2, 0, 1))
                    np_sr2_pos.append(sr2_pieces.transpose(2, 0, 1))
                    np_label_pos.append(label_change)
        return np_sr1_pos, np_sr1_neg, np_sr2_pos, np_sr2_neg, np_label_pos, np_label_neg
    else:
        # program only support 3 channels now!
        print('Not support numbers of chanel except 1 and 3!')
        sys.exit(-1)


def normalize(image, mean, std):
    """
    # Normalize an image
    :param image: image.shape: (h,w,3)
    :param mean: Given mean, mean.shape:(1,3)
    :param std: Given std, std.shape(1,3)
    :return:
    """
    image = image.astype('float')
    if image.shape[2] is not 3:
        print('Image must contains three channels!')
        sys.exit()
    else:
        for i in range(3):
            channel = image[:, :, i]
            image[:, :, i] = (channel - mean[i]) / std[i]
        return image


class MyData0518(Dataset):
    def __init__(self, txt_path):
        """
        :param txt_path: Contain the data path.
        """
        file = open(txt_path, 'r')
        # record the numbers of pos and neg samples
        first_line = file.readline()
        mean_std = first_line.split(':')
        mean = mean_std[1:4]                    # get the mean, mean.type=char
        std = mean_std[5:8]                     # get the std, std.type=char
        self.mean = [float(x) for x in mean]         # convert the str into float
        self.std = [float(x) for x in std]
        self.sr1 = []
        self.sr2 = []
        self.label = []
        # record the length of pos and neg samples
        for line in file:
            words = line.split()
            sr1 = words[0]      # get the sr1 image
            sr2 = words[1]      # get the sr2 image
            label = words[2]    # get the label
            self.sr1.append(sr1)
            self.sr2.append(sr2)
            self.label.append(label)
        self.len = len(self.sr1)
        print('Completely load images')

    def __getitem__(self, item):
        sr1 = cv2.imread(self.sr1[item])
        sr2 = cv2.imread(self.sr2[item])
        label = cv2.imread(self.label[item], cv2.IMREAD_GRAYSCALE)
        sr1 = normalize(sr1, self.mean, self.std)
        sr2 = normalize(sr2, self.mean, self.std)
        return sr1.transpose(2, 0, 1), sr2.transpose(2, 0, 1), label/255

    def __len__(self):
        return self.len

    def __get_label_name__(self, item):
        return self.label[item]


class MyDataTest(Dataset):
    def __init__(self, txt, stride, height_size, width_size, mean, std):
        if stride is None or stride == 0:
            # stride can not be None!
            print('Stride can not be None or zero!')
            sys.exit(-1)
        if width_size is None or height_size is None:
            # width or height size can not be None!
            print('width or height size can not be None!')
            sys.exit(-1)
        words = txt.split(' ')
        sr1 = normalize(cv2.imread(words[0]), mean, std)
        sr2 = normalize(cv2.imread(words[1]), mean, std)
        self.sr1 = []
        self.sr2 = []
        h, w, c = sr1.shape                                         # get the shape
        self.shape = (h, w)                                         # record the original shape
        height_steps = math.ceil((h - height_size) / stride + 1)
        wide_steps = math.ceil((w - width_size) / stride + 1)
        self.steps = (height_steps, wide_steps)                     # record steps each dim
        if wide_steps is 0 or height_steps is 0:
            print('Error, this is because stride equals 1 and image size is one larger than output size.')
            sys.exit(-1)
        if c == 3:
            height_fill = (height_steps - 1) * stride + height_size - h  # The number of pixels to fill in the height
            wide_fill = (wide_steps - 1) * stride + width_size - w  # The number of pixels to fill in the width
            # fill the border
            sr1 = cv2.copyMakeBorder(sr1, 0, height_fill, 0, wide_fill, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            sr2 = cv2.copyMakeBorder(sr2, 0, height_fill, 0, wide_fill, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            np_sr1 = []  # save the result of cut
            np_sr2 = []
            for i in range(height_steps):
                for j in range(wide_steps):
                    sr1_pieces = sr1[i * stride:i * stride + height_size, j * stride:j * stride + width_size, :]
                    sr2_pieces = sr2[i * stride:i * stride + height_size, j * stride:j * stride + width_size, :]
                    np_sr1.append(sr1_pieces.transpose(2, 0, 1))
                    np_sr2.append(sr2_pieces.transpose(2, 0, 1))
            self.sr1 = np_sr1
            self.sr2 = np_sr2
            self.len = len(np_sr1)
            print('Completely cut a pair of images')
        else:
            # program only support 3 channels now!
            print('Not support numbers of chanel except 3!')
            sys.exit(-1)

    def __getitem__(self, item):
        return self.sr1[item], self.sr2[item]

    def __len__(self):
        return self.len


train_txt = 'train.txt'


if __name__ == '__main__':
    train = MyData0518(train_txt)
    train_loader = DataLoader(train, batch_size=20, shuffle=True, drop_last=True)
    print(train.len)
    for index, (sr1, sr2, label) in enumerate(train_loader):
        print(sr1.shape)
        print(torch.mean(sr1), torch.std(sr1))
        print(torch.max(label))
        sys.exit(0)
    pass
