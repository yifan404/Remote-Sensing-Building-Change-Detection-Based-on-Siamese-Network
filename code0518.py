import os
import torch
import numpy as np
import torch.optim as optim
import data_loader
import siamunet_conc
from torch.utils.data import DataLoader
import math
import cv2
import sys
import platform
import random
"""
# 分割图片，生成地址索引，分割数据集，训练，测试，输出预测结果，主函数
# 作者：刘一帆
# 日期：2020/5/19
# 结果：完成。
"""


def train():
    my_model.train()
    pre_loss = 1
    for epoch in range(iteration):
        for index, (sr1, sr2, label) in enumerate(train_loader):
            loss_average = []
            optimizer.zero_grad()
            fcn_result = my_model(sr1.float().cuda(), sr2.float().cuda())
            loss = loss_function(fcn_result.float(), label.long().cuda())
            loss.backward()
            optimizer.step()
            cur_loss = loss.cpu().detach().numpy()
            loss_average.append(cur_loss)
            if index % 20 == 0:
                # Print loss information every 20 times
                cur_loss = np.mean(loss_average)
                print('\r {}/{}, loss average is :{}'.format(epoch, iteration, cur_loss), end='')
                loss_average = []
            if cur_loss < pre_loss:
                pre_loss = cur_loss
                torch.save(my_model, model_name)
                print(' ')
                print('Model save completely with loss:{} at {}th epoch!'.format(cur_loss, epoch))
                # evaluation()


def evaluation():
    """
    # evaluate the whole specified data sets.
    :return:
    """
    # my_model.eval()
    print("Evaluating!")
    final_result = np.zeros((validation_data.__len__(), w_size, h_size))
    final_label = np.zeros((validation_data.__len__(), w_size, h_size))
    for index, (sr1, sr2, label) in enumerate(validation_loader):
        fcn_result = my_model(sr1.float().cuda(), sr2.float().cuda())
        output_np = np.argmax(fcn_result.cpu().detach().numpy(), axis=1)
        final_result[index * batch_size:index * batch_size + output_np.shape[0], :, :] = output_np
        final_label[index * batch_size: index * batch_size + output_np.shape[0], :, :] = label
    pixel_num = final_result.size                                                         # total pixel numbers
    wrong_pixel_num = np.sum(final_label + final_result == 1)                             # wrong pixel numbers
    right_pixel_num = pixel_num - wrong_pixel_num                                         # right pixel numbers
    right_rate = right_pixel_num / pixel_num                                              # accuracy rate
    print('**************************************')
    print('Overall Accuracy of evaluation (OA): {:.2%}'.format(right_rate))
    change_detect = np.sum(final_result * final_label == 1)                               # label 1, prediction 1
    change_not_detect = np.sum((final_result + 1) * final_label == 1)                     # label 1, prediction 0
    not_change_detect = wrong_pixel_num - change_not_detect                               # label 0, prediction 1
    not_change_not_detect = right_pixel_num - change_detect                               # label 0, prediction 0
    print("True Positive (TP)：%.2f" % (100 * change_detect / pixel_num), '%')
    print("True Negative (TN)：%.2f" % (100 * not_change_not_detect / pixel_num), '%')
    print("False Negative (FN)：%.2f" % (100 * change_not_detect / pixel_num), '%')
    print("False Positive (FP)：%.2f" % (100 * not_change_detect / pixel_num), '%')
    precision = change_detect / (change_detect + not_change_detect)
    print("Precision：%.2f" % (100 * precision), '%')
    recall = change_detect / np.sum(final_label == 1)
    print("Recall：%.2f" % (100 * recall), '%')
    print("F1 score：%.2f" % (100 * 2 * precision * recall / (precision + recall)), '%')
    print("Evaluate completely!")


def test():
    """
    # output test images
    :return:
    """
    print("Predicting!")
    my_model.eval()
    f_np = np.zeros((test_data.len, 512, 512))
    for index, (sr1, sr2, label) in enumerate(test_loader):
        if sr1.shape[0] == batch_size:
            fcn_result = my_model(sr1.float().cuda(), sr2.float().cuda())
            output_np = fcn_result.cpu().detach().numpy()
            output_np = np.argmax(output_np, axis=1)
            f_np[index * batch_size:index * batch_size + output_np.shape[0], :, :] = output_np
        else:
            keep_len = sr1.shape[0]
            double_numbers = math.ceil((batch_size - keep_len) / keep_len)
            sr1.repeat((double_numbers, 0, 0, 0))
            sr2.repeat((double_numbers, 0, 0, 0))
            fcn_result = my_model(sr1[:batch_size, :, :, :].float().cuda(), sr2[:batch_size, :, :, :].float().cuda())
            output_np = fcn_result.cpu().detach().numpy()
            output_np = np.argmax(output_np, axis=1)
            f_np[index * batch_size:index * batch_size + keep_len, :, :] = output_np
    f_np = 255 * f_np.transpose(1, 2, 0)
    print("Predict completely!")
    print("Writing images!")
    for i in range(test_data.len):
        full_name = test_data.__get_label_name__(i)
        file_name = os.path.basename(full_name)
        name = file_name.split('.')
        cv2.imwrite(prediction_path + '/' + str(name[0]) + 'pre' + str(i) + '.tif', f_np[:, :, i].astype('int'))
    print('Write over')


def mean_std_count(image):
    """
    # Count the mean and std of each channel of image
    :param image: original image
    :return: mean: numpy array; shape=(1,3), std: numpy array; shape=(1,3)
    """
    mean = np.zeros((1, 3))
    std = np.zeros((1, 3))
    for i in range(3):
        c = image[:, :, i]
        mean[0, i] = np.mean(c)
        std[0, i] = np.std(c)
    return mean, std


def image_mean_std_write(txt, length):
    """
    # Write the mean and std at the first line of the txt file.
    :param txt: txt file
    :param length: The number of paris of images
    :return: 0
    """
    with open(txt, 'r+') as file:
        original = file.read()
        file.seek(0)
        mean = np.zeros((2 * length, 3))
        std = np.zeros((2 * length, 3))
        time = 0
        for line in file:
            words = line.split()
            sr1 = cv2.imread(words[0])
            mean_temp, std_temp = mean_std_count(sr1)
            mean[time, :] = mean_temp
            std[time, :] = std_temp
            time += 1
            sr2 = cv2.imread(words[1])
            mean_temp, std_temp = mean_std_count(sr2)
            mean[time, :] = mean_temp
            std[time, :] = std_temp
            time += 1
        file.seek(0)
        mean = np.mean(mean, axis=0)
        std = np.mean(std, axis=0)
        file.write('mean:')
        for i in range(3):
            file.write(str(mean[i]))
            file.write(':')
        file.write('std:')
        for i in range(3):
            file.write(str(std[i]))
            file.write(':')
        file.write('\n')
        file.write(original)
        file.close()
    return 0


def sr1_sr2_label_path_make(sr1_path, sr2_path, label_path, txt_name, system_platform=None, mean_std=False):
    """
    :param sr1_path: sr1 path
    :param sr2_path: sr2 path
    :param label_path: label path
    :param txt_name: file name you need
    :param system_platform: if none, the program automatically selects the format that fit the current system.
        if you specify the format ahead of time, program will check weather support it or not.
    :param mean_std: At the first line, write mean and std of each channel in sr1 and sr2 images or not.
        I highly recommend use it as long as you keep using my code.
    :return: None
    """
    if os.path.exists(txt_name):
        # Check the file exits or not
        print(txt_name, "exists! Do you want to cover it ?")
        while True:
            # File exits! program ask user to choose!
            print("Input 'y' to cover it or 'n' to quit system!")
            switch = input("Please input: ")
            if switch == 'n':
                # User do not want to cover the original file.
                sys.exit(0)
            elif switch == 'y':
                # User decide to empty and rewrite the original file.
                txt_file = open(txt_name, 'w')
                txt_file.seek(0)
                txt_file.truncate()
                break
            else:
                # User input a key except for 'y' and 'n'.
                print('No such choices! Try again!')
    else:
        # file do not exits.
        txt_file = open(txt_name, 'w')
    if system_platform is None:
        # User do not specify the system ahead of time. System will choose automatically.
        system_platform = platform.system()
        if system_platform == "Windows":
            print('Windows system!')
        elif system_platform == "Linux":
            print('Linux system!')
        else:
            print('Not support current system!')
            sys.exit(-1)
    if system_platform == 'Windows':
        for i in range(len(os.listdir(sr1_path))):
            sr1_file = os.listdir(sr1_path)
            sr1_file.sort()
            sr2_file = os.listdir(sr2_path)
            sr2_file.sort()
            label_file = os.listdir(label_path)
            label_file.sort()
            txt_file.write(os.path.join(sr1_path, sr1_file[i]))
            txt_file.write(' ')
            txt_file.write(os.path.join(sr2_path, sr2_file[i]))
            txt_file.write(' ')
            txt_file.write(os.path.join(label_path, label_file[i]))
            if (i + 1) == len(os.listdir(sr1_path)):
                # The last line do not need new line.
                break
            else:
                txt_file.write('\n')
    elif system_platform == 'Linux':
        for i in range(len(os.listdir(sr1_path))):
            sr1_file = os.listdir(sr1_path)
            sr1_file.sort()
            sr2_file = os.listdir(sr2_path)
            sr2_file.sort()
            label_file = os.listdir(label_path)
            label_file.sort()
            txt_file.write(sr1_path + '/' + sr1_file[i])
            txt_file.write(' ')
            txt_file.write(sr2_path + '/' + sr2_file[i])
            txt_file.write(' ')
            txt_file.write(label_path + '/' + label_file[i])
            if (i + 1) == len(os.listdir(sr1_path)):
                # The last line do not need new line.
                break
            else:
                txt_file.write('\n')
    else:
        # User specify a system that program do not support currently.
        print('Not support ', system_platform, ' system currently!')
        txt_file.close()
        sys.exit(1)
    # save the file
    txt_file.close()
    print('file write over!')
    print('Counting mean & std!')
    if mean_std is True:
        # Add the mean and std at the first line.
        image_mean_std_write(txt_name, len(os.listdir(sr1_path)))
    print("Write over!")
    print('Mean & std write over!')
    return 0


def train_validation_test_division(all_file, train_file, validation_file, test_file,
                                   train_percentage, validation_percentage, test_percentage):
    """
    # divide all file into train, validation and test file.
    :param all_file: all data
    :param train_file: train file name
    :param validation_file: validate file name
    :param test_file: test file name
    :param train_percentage: train percentage which ranges from 0 to 1
    :param validation_percentage: validation percentage which ranges from train_percentage to 1
    :param test_percentage: test percentage which ranges from validation_percentage to 1
    :return:
    """
    with open(all_file, 'r') as file:
        first_line = file.readline()                    # read the mean and std value
        first_line = first_line.strip('\n')             # remove '\n'
        train = open(train_file, 'w+')                  # create train file
        validation = open(validation_file, 'w+')        # create validation file
        test = open(test_file, 'w+')                    # create test file
        train.write(first_line)                         # write mean and std value into every file
        validation.write(first_line)
        test.write(first_line)
        for line in file:
            i = random.random()                         # generate a random number
            line = line.strip('\n')                     # remove '\n' of the end
            if i < train_percentage:
                train.write('\n')
                train.write(line)
            elif i < validation_percentage:
                validation.write('\n')
                validation.write(line)
            elif i < test_percentage:
                test.write('\n')
                test.write(line)
        train.close()                                   # close file
        validation.close()
        test.close()


def cut_image_into_pieces(sr1, sr2, label, sr1_path, sr2_path,
                          label_path, stride=None, width_size=None, height_size=None):
    """
    # Cut the picture into small pieces and divide into two categories
    :param sr1: sr1
    :param sr2: sr2
    :param label: label
    :param sr1_path: storage path of sr1
    :param sr2_path: storage path of sr2
    :param label_path: storage path of label
    :param stride: width and height stride, it can not be zero.
    :param width_size: expected image size in the width.
    :param height_size: expected image size in the height
    :return:
    """
    print('Loading images!')
    sr1 = cv2.imread(sr1)
    sr2 = cv2.imread(sr2)
    label = cv2.imread(label)
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
        print('Cutting images!')
        for i in range(height_steps):
            for j in range(wide_steps):
                label_change = label[i * stride:i * stride + height_size, j * stride:j * stride + width_size]
                sr1_pieces = sr1[i * stride:i * stride + height_size, j * stride:j * stride + width_size, :]
                sr2_pieces = sr2[i * stride:i * stride + height_size, j * stride:j * stride + width_size, :]
                cv2.imwrite(sr1_path + '/' + str(i) + '_' + str(j) + '.tif', sr1_pieces)
                cv2.imwrite(sr2_path + '/' + str(i) + '_' + str(j) + '.tif', sr2_pieces)
                cv2.imwrite(label_path + '/' + str(i) + '_' + str(j) + '.tif', label_change)
        print('Cut completely!')
    else:
        # program only support 3 channels now!
        print('Not support numbers of chanel except 1 and 3!')
        sys.exit(-1)


if __name__ == '__main__':
    """
    # define some parameters 
    """
    all_txt = 'all.txt'                                 # record all files
    train_txt = 'train.txt'                             # train data sets
    validation_txt = 'validation.txt'                   # evaluation data sets
    test_txt = 'test.txt'                               # test data sets
    """
    # source image
    """
    sr1_image = './data/before/before.tif'
    sr2_image = './data/after/after.tif'
    label_image = './data/change label/change_label.tif'
    """
    # output folder
    """
    sr1_cut_path = './data/sr1_cut'
    sr2_cut_path = './data/sr2_cut'
    label_cut_path = './data/label_cut'
    prediction_path = './data/pre'

    if os.path.exists(sr1_cut_path):
        print('sr1 cut path exits!')
    else:
        os.mkdir(sr1_cut_path)

    if os.path.exists(sr2_cut_path):
        print('sr2 cut path exits!')
    else:
        os.mkdir(sr2_cut_path)

    if os.path.exists(label_cut_path):
        print('label cut path exits!')
    else:
        os.mkdir(label_cut_path)

    if os.path.exists(prediction_path):
        print('prediction path exits!')
    else:
        os.mkdir(prediction_path)

    """
    # model parameters
    """
    model_name = 'Model0518_siamase.pkl'                # model name
    iteration = 1000                                    # times of train iterations
    batch_size = 2                                      # batch size
    w_size = 512                                        # image size
    h_size = 512
    stride = 512                                        # window slide steps(pixel)
    my_model = siamunet_conc.SiamUnet_conc(input_nbr=3, label_nbr=2).cuda()    # model
    weight = torch.from_numpy(np.array([1, 1]))         # weights of CrossEntropyLoss
    loss_function = torch.nn.CrossEntropyLoss(weight=weight.float().cuda()).cuda()
    optimizer = optim.Adam(my_model.parameters(), lr=1e-4)
    if os.path.exists(model_name):
        # If model exits already, load it.
        my_model = torch.load(model_name).cuda()
        print('Model exits!')
    else:
        print('Create new model!')
    while True:
        print('********************Function choose!***********************')
        print("1.Input 'c' to cut images into pieces!")
        print("2.Input 'w' to write all image paths!")
        print("3.Input 'd' to divide data set into 'train set', 'validation set', 'test set' !")
        print("4.Input 't' to train the model! ")
        print("5.Input 'v' to validation!")
        print("6.Input 'o' to output test image!")
        print("7.Input 'q' to quit program!")
        print("**********************************************************")
        switch = input('Please input here:')
        if switch == 't':
            train_data = data_loader.MyData0518(train_txt)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            train()
            break       # quit the program, or delete this line for not quiting after training
        elif switch == 'v':
            print("-----------------")
            print("Data sets choose!")
            print("Input 1 to evaluate the validation data set or 2 to evaluate the test data set!")
            ch = input('Input here:')
            if ch == '1':
                validation_data = data_loader.MyData0518(validation_txt)
                validation_loader = DataLoader(validation_data, batch_size=batch_size, drop_last=False)
                evaluation()
            elif ch == '2':
                validation_data = data_loader.MyData0518(test_txt)
                validation_loader = DataLoader(validation_data, batch_size=batch_size, drop_last=False)
                evaluation()
            else:
                print("No such choice! Try again please!")
        elif switch == 'o':
            test_data = data_loader.MyData0518(test_txt)
            test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=False, shuffle=False)
            test()
            break
        elif switch == 'c':
            cut_image_into_pieces(sr1_image, sr2_image, label_image, sr1_cut_path, sr2_cut_path, label_cut_path,
                                  stride=stride, width_size=w_size, height_size=h_size)
        elif switch == 'w':
            sr1_sr2_label_path_make(sr1_cut_path, sr2_cut_path, label_cut_path, all_txt, mean_std=True)
        elif switch == 'd':
            train_validation_test_division(all_txt, train_txt, validation_txt, test_txt, 0.7, 0.8, 1)
        elif switch == 'q':
            break
        else:
            print("No such function! Try again!")
