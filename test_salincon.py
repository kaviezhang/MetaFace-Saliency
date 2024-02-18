import os
import torch
import cv2
import numpy as np
from torchvision import transforms, utils, models
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot, image


def draw_display(dispsize, imagefile=None):
    # construct screen (black background)
    screen = np.zeros((dispsize[1], dispsize[0], 3), dtype='float32')
    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        # img = image.imread(imagefile)
        img = cv2.imread(imagefile)
        img = cv2.resize(img, dispsize)
        src = cv2.imread(imagefile, 0)
        src = cv2.resize(src, dispsize)
        index = np.where(src == 255)

        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = dispsize[0] // 2 - w // 2
        y = dispsize[1] // 2 - h // 2
        # draw the image on the screen
        screen[y:y + h, x:x + w, :] += img
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    screen = screen.astype(np.uint8)  # img是uint8的ndarray
    ax.imshow(screen)  # , origin='upper')
    return fig, ax, index


transform = transforms.Compose([
    transforms.Resize((288, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

flag = 9  # 0 for TranSalNet_Dense, 1 for TranSalNet_Res

if flag == 5:
    # from liif_model.liif import *
    # model = LIIF()
    from Model import *
    model = UNet()
if flag == 0:
    from TranSalNet_Res import TranSalNet
    model = TranSalNet()
    # model.load_state_dict(torch.load(r'transrespretrain0429.pth'))
if flag == 2:
    # from liif_model.vgg import *
    # model = B2_VGG()
    from Model import *
    model = Deconv()
    # model.load_state_dict(torch.load(r'mymodel.pth'))
if flag == 1:
    from TranSalNet_Dense import TranSalNet
    model = TranSalNet()
    # model.load_state_dict(torch.load(r'transdensepretrain0403.pth'))
if flag == 3:
    from liif_model.TASED import TASED_v2
    model = TASED_v2()
if flag == 4:
    from liif_model.SIMPLEnet import *
    # model = PNASModel()
    # model = DenseModel()
    # model = ResNetModel()
    model = VGGModel()
    # model = MobileNetV2()
    # model.load_state_dict(torch.load(r'mymodel.pth'))
if flag == 6:
    from liif_model.salnet import salnet
    model = salnet()
if flag == 7:
    from PbMoble import *
    model = Deconv()
if flag == 9:
    from HATModel import *
    model = Swin_cpas_offset()
if flag == 10:
    from HyperAttention import *
    model = HyperAttention()

model.load_state_dict(torch.load(r'mymodel_salicon.pth'))
model = model.to(device)
model.eval()

if __name__ == '__main__':
    path = "D:/workspace/python/unisal-master/examples/views/images"
    path_test = "D:/workspace/python/unisal-master/examples/views/saliency"
    list_paths = os.listdir(path)
    for img_name in list_paths:
        img_path = os.path.join(path, img_name)
        # img = preprocess_img(img_path)  # padding and resizing input image into 384x288
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        img = img.type(torch.cuda.FloatTensor).to(device)
        img = img.unsqueeze(0)
        pred_saliency = model(img)
        pred = pred_saliency.cpu().detach().numpy()
        output = np.clip(pred * 255, 0, 255)
        output = output[0, 0, :, :]
        # toPIL = transforms.ToPILImage()
        # pic = toPIL(pred_saliency.squeeze())
        # pred_saliency = postprocess_img(pic, img_path)  # restore the image to its original size as the result
        dispsize = (384, 288)

        fig, ax, index = draw_display(dispsize, imagefile=img_path)
        src = cv2.imread(img_path, 0)
        # index = np.where(src == 255)
        # lowbound = np.mean(output[output > 0])
        # output[output < lowbound] = np.NaN
        lowbound = np.mean(output[output > 0])
        output[output < lowbound] = lowbound/2
        output[index] = np.NaN
        ax.imshow(output, cmap='jet', alpha=0.5)
        test_path = os.path.join(path_test, img_name)
        savefilename = str(test_path)
        ax.invert_yaxis()
        fig.savefig(savefilename)
        # test_path = os.path.join(path_test, img_name)
        # cv2.imwrite(test_path, output, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # save the result
        # print('Finished, check the result at: {}'.format(img_name))
        # print()



