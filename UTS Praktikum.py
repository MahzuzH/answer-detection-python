import math
import sys
import cv2
import pandas
import numpy as np
import utlis
import imutils
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PIL import ImageQt
from matplotlib import pyplot as plt


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUITubes.ui', self)
        self.Image = None
        self.Image2 = None
        self.hasil = None
        self.button_loadcitra.clicked.connect(self.fungsi)
        self.button_savecitra.clicked.connect(self.save)
        self.actionGrayscal.triggered.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.contrastStretching)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBiner.triggered.connect(self.biner)
        self.actionHistogram_Grayscale_2.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB_2.triggered.connect(self.RGBHistogram)
        self.actionHistogram_Equalization_3.triggered.connect(self.EqualHistogram)
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action_45_Derajat.triggered.connect(self.rotasimin45)
        self.action45_Derajat.triggered.connect(self.rotasi45)
        self.action_90_Derajat.triggered.connect(self.rotasimin90)
        self.action90_Derajat.triggered.connect(self.rotasi90)
        self.action180_Derajat.triggered.connect(self.rotasi180)
        self.actionTranspose.triggered.connect(self.transpose)
        self.action2x.triggered.connect(self.ZoomIn2x)
        self.action3x.triggered.connect(self.ZoomIn3x)
        self.action4x.triggered.connect(self.ZoomIn4x)
        self.action1_2x.triggered.connect(self.ZoomOut2x)
        self.action1_3x.triggered.connect(self.ZoomOut3x)
        self.action1_4x.triggered.connect(self.ZoomOut4x)
        self.action480P.triggered.connect(self.SD)
        self.action720P.triggered.connect(self.HD)
        self.action1080P.triggered.connect(self.FHD)
        self.actionCrop.triggered.connect(self.crop)
        self.actionTambah.triggered.connect(self.aritmatikaTambah)
        self.actionKurang.triggered.connect(self.aritmatikaKurang)
        self.actionKali.triggered.connect(self.aritmatikaKali)
        self.actionBagi.triggered.connect(self.aritmatikaBagi)
        self.actionAND.triggered.connect(self.operasiAnd)
        self.actionOR.triggered.connect(self.operasiOr)
        self.actionXOR.triggered.connect(self.operasiXor)
        self.actionKernel_1.triggered.connect(self.konvolusikernel1)
        self.actionKernel_6.triggered.connect(self.konvolusikernel6)
        self.action3X3.triggered.connect(self.mean3x3)
        self.action2X2.triggered.connect(self.mean2x2)
        self.actionGauss.triggered.connect(self.gauss)
        self.actioni.triggered.connect(self.sharpeningi)
        self.actionii.triggered.connect(self.sharpeningii)
        self.actioniii.triggered.connect(self.sharpeningiii)
        self.actioniv.triggered.connect(self.sharpeningiv)
        self.actionv.triggered.connect(self.sharpeningv)
        self.actionvi.triggered.connect(self.sharpeningvi)
        self.actionLaplace.triggered.connect(self.sharpeninglaplace)
        self.actionMedian.triggered.connect(self.median)
        self.actionMaxFilter.triggered.connect(self.maxFilter)
        self.actionMinFilter.triggered.connect(self.minFilter)
        self.actionDFT_Smooth.triggered.connect(self.DFT_Smooth)
        self.actionDFT_Edge.triggered.connect(self.DFT_Edge)
        self.actionCanny_Edge_Detection_2.triggered.connect(self.CannyEdge)
        self.actionCountour.triggered.connect(self.Countour)
        self.pushButton.clicked.connect(self.run)
        self.printpiksel.clicked.connect(self.printpik)

    def printpik(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save Piksel', '', 'Excel(*.xlsx)')

        if path == "":
            return

        exc = pandas.DataFrame(self.hasil)
        exc.to_excel(path)

    def save(self):
        filePath, _ = QFileDialog.getSaveFileName()

        if filePath == "":
            return

        image = ImageQt.fromqpixmap(self.label_2.pixmap())
        image.save(filePath)

    def fungsi(self):
        imagePath, _ = QFileDialog.getOpenFileName()
        self.Image = cv2.imread(imagePath)
        pixmap = QPixmap(imagePath)
        self.label.setPixmap(pixmap)
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label.setScaledContents(True)
        self.displayImage(1)
        img = self.Image
        path, _ = QFileDialog.getSaveFileName(self, 'Save Piksel', '', 'Excel(*.xlsx)')

        if path == "":
            return

        exc = pandas.DataFrame(img)
        exc.to_excel(path)

    def grayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i,j] = np.clip(0.299 * self.Image[i,j,0] +
                                    0.587 * self.Image[i,j,1] +
                                    0.114 * self.Image[i,j,2], 0, 255)
        self.Image = gray
        self.hasil = gray
        self.displayImage(2)

        print(gray)

    def brightness (self):
        try:
            H, W = self.Image.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                         0.587 * self.Image[i, j, 1] +
                                         0.114 * self.Image[i, j, 2], 0, 255)
            self.Image = gray
        except:
            pass

        H, W = self.Image.shape[:2]
        brightness = 30
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i,j)
                b = np.clip(a + brightness, 0, 255)
                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0
                else:
                    b = b
                self.Image.itemset((i,j), b)
        bright = self.Image
        self.hasil = bright
        self.displayImage(2)
        print(self.Image)

    def contrast (self):
        try:
            H, W = self.Image.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                         0.587 * self.Image[i, j, 1] +
                                         0.114 * self.Image[i, j, 2], 0, 255)
            self.Image = gray
        except:
            pass

        H, W = self.Image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i,j)
                b = math.ceil(a * contrast)
                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0
                else:
                    b = b
                self.Image.itemset((i,j), b)
        cont = self.Image
        self.hasil = cont
        self.displayImage(2)
        print(self.Image)

    def contrastStretching(self):
        try:
            H, W = self.Image.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                         0.587 * self.Image[i, j, 1] +
                                         0.114 * self.Image[i, j, 2], 0, 255)
            self.Image = gray
        except:
            pass

        H, W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i,j)
                b = float(a - minV) / (maxV - minV) * 255
                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0
                else:
                    b = b
                self.Image.itemset((i,j), b)
        contsctrech = self.Image
        self.hasil = contsctrech
        self.displayImage(2)
        print(self.Image)

    def negative(self):
        try:
            H, W = self.Image.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                         0.587 * self.Image[i, j, 1] +
                                         0.114 * self.Image[i, j, 2], 0, 255)
            self.Image = gray
        except:
            pass

        H, W = self.Image.shape[:2]
        negative = 255

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = math.ceil(negative - a)

                self.Image.itemset((i, j), b)
        neg = self.Image
        self.hasil = neg
        self.displayImage(2)
        print(self.Image)

    def biner(self):
        try:
            H, W = self.Image.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                         0.587 * self.Image[i, j, 1] +
                                         0.114 * self.Image[i, j, 2], 0, 255)
            self.Image = gray
        except:
            pass

        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                if a < 180:
                    b = 1
                elif a > 180:
                    b = 255
                else:
                    b = 0

                self.Image.itemset((i, j), b)
        bin = self.Image
        self.hasil = bin
        self.displayImage(2)

    def grayHistogram(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i,j] = np.clip(0.299 * self.Image[i,j,0] +
                                    0.587 * self.Image[i,j,1] +
                                    0.114 * self.Image[i,j,2], 0, 255)
        self.Image = gray
        self.displayImage(2)
        print(gray)
        plt.hist(self.Image.ravel(), 255, [0, 255])
        plt.show()

    def RGBHistogram(self):
        color = ("b","g","r")
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.Image],[i], None, [256], [0,256])
            plt.plot(histo, color=col)
            plt.xlim([0, 256])
        self.displayImage(2)
        plt.show()

    def EqualHistogram(self):
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalize = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype("uint8")
        self.Image = cdf[self.Image]
        self.displayImage(2)

        plt.plot(cdf_normalize, color="b")
        plt.hist(self.Image.flatten(), 256, [0, 256], color="r")
        plt.xlim([0, 256])
        plt.legend(("cdf","histogram"), loc="upper left")
        plt.show()

    def translasi(self):
        h, w = self.Image.shape[:2]
        quarter_h, quarter_w = h/4,w/4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.Image, T, (w, h))
        self.Image = img
        self.displayImage(2)
        print(self.Image)

    def rotasimin45(self):
        self.rotasi(315)

    def rotasi45(self):
        self.rotasi(40)

    def rotasimin90(self):
        self.rotasi(270)

    def rotasi90(self):
        self.rotasi(90)

    def rotasi180(self):
        self.rotasi(180)

    def transpose(self):
        img = cv2.transpose(self.Image)
        self.Image = img
        self.displayImage(2)

    def rotasi(self, degree):
        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 0.7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2

        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h, w))
        self.Image = rot_image
        self.displayImage(2)

    def ZoomIn2x(self):
        skala = 2
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def ZoomIn3x(self):
        skala = 3
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def ZoomIn4x(self):
        skala = 4
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def ZoomOut2x(self):
        skala = 1/2
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom Out', resize_image)
        cv2.waitKey()

    def ZoomOut3x(self):
        skala = 1/4
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom Out', resize_image)
        cv2.waitKey()

    def ZoomOut4x(self):
        skala = 3/4
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom Out', resize_image)
        cv2.waitKey()

    def SD(self):
        x = 852
        y = 480
        res = (x, y)
        resize_img = cv2.resize(self.Image, res, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('480P', resize_img)
        cv2.waitKey()

    def HD(self):
        x = 1280
        y = 720
        res = (x, y)
        resize_img = cv2.resize(self.Image, res, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('720P', resize_img)
        cv2.waitKey()

    def FHD(self):
        x = 1920
        y = 1080
        res = (x, y)
        resize_img = cv2.resize(self.Image, res, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('1080P', resize_img)
        cv2.waitKey()

    def crop(self):
        H, W = self.Image.shape[:2]
        start_row = H-600
        end_row = H
        start_cols = W-200
        end_cols = W
        crop = self.Image[start_row:end_row, start_cols:end_cols]
        cv2.imshow('Original', self.Image)
        cv2.imshow('Cropped', crop)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def aritmatikaTambah(self):
        imagePath1, _ = QFileDialog.getOpenFileName()
        self.Image = cv2.imread(imagePath1)
        imagePath2, _ = QFileDialog.getOpenFileName()
        self.Image2 = cv2.imread(imagePath2)

        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        H, W = self.Image2.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image2[i, j, 0] +
                                     0.587 * self.Image2[i, j, 1] +
                                     0.114 * self.Image2[i, j, 2], 0, 255)
        self.Image2 = gray
        Image1 = self.Image
        Image2 = self.Image2

        ImageTambah = Image1 + Image2
        self.hasil = ImageTambah

        cv2.imshow("Image 1 Original", Image1)
        cv2.imshow("Image 2 Original", Image2)
        cv2.imshow("Image Tambah", ImageTambah)

    def aritmatikaKurang(self):
        imagePath1, _ = QFileDialog.getOpenFileName()
        self.Image = cv2.imread(imagePath1)
        imagePath2, _ = QFileDialog.getOpenFileName()
        self.Image2 = cv2.imread(imagePath2)

        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        H, W = self.Image2.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image2[i, j, 0] +
                                     0.587 * self.Image2[i, j, 1] +
                                     0.114 * self.Image2[i, j, 2], 0, 255)
        self.Image2 = gray
        Image1 = self.Image
        Image2 = self.Image2

        ImageKurang = Image1 - Image2
        self.hasil = ImageKurang

        cv2.imshow("Image 1 Original", Image1)
        cv2.imshow("Image 2 Original", Image2)
        cv2.imshow("Image Kurang", ImageKurang)

    def aritmatikaKali(self):
        imagePath1, _ = QFileDialog.getOpenFileName()
        self.Image = cv2.imread(imagePath1)
        imagePath2, _ = QFileDialog.getOpenFileName()
        self.Image2 = cv2.imread(imagePath2)

        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        H, W = self.Image2.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image2[i, j, 0] +
                                     0.587 * self.Image2[i, j, 1] +
                                     0.114 * self.Image2[i, j, 2], 0, 255)
        self.Image2 = gray
        Image1 = self.Image
        Image2 = self.Image2

        ImageKali = Image1 * Image2
        self.hasil = ImageKali

        cv2.imshow("Image 1 Original", Image1)
        cv2.imshow("Image 2 Original", Image2)
        cv2.imshow("Image Kali", ImageKali)


    def aritmatikaBagi(self):
        imagePath1, _ = QFileDialog.getOpenFileName()
        self.Image = cv2.imread(imagePath1)
        imagePath2, _ = QFileDialog.getOpenFileName()
        self.Image2 = cv2.imread(imagePath2)

        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        H, W = self.Image2.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image2[i, j, 0] +
                                     0.587 * self.Image2[i, j, 1] +
                                     0.114 * self.Image2[i, j, 2], 0, 255)
        self.Image2 = gray
        Image1 = self.Image
        Image2 = self.Image2

        ImageBagi = Image1 / Image2
        self.hasil = ImageBagi

        cv2.imshow("Image 1 Original", Image1)
        cv2.imshow("Image 2 Original", Image2)
        cv2.imshow("Image Bagi", ImageBagi)

        path, _ = QFileDialog.getSaveFileName(self, 'Save Piksel', '', 'Excel(*.xlsx)')

        if path == "":
            return

        exc = pandas.DataFrame(ImageBagi)
        exc.to_excel(path)

    def operasiAnd(self):
        imagePath1, _ = QFileDialog.getOpenFileName()
        self.Image = cv2.imread(imagePath1)
        imagePath2, _ = QFileDialog.getOpenFileName()
        self.Image2 = cv2.imread(imagePath2)

        Image1 = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        Image2 = cv2.cvtColor(self.Image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_and(Image1, Image2)

        cv2.imshow("Image 1 Original", Image1)
        cv2.imshow("Image 2 Original", Image2)
        cv2.imshow("Image Operasi AND", operasi)

    def operasiOr(self):
        imagePath1, _ = QFileDialog.getOpenFileName()
        self.Image = cv2.imread(imagePath1)
        imagePath2, _ = QFileDialog.getOpenFileName()
        self.Image2 = cv2.imread(imagePath2)

        Image1 = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        Image2 = cv2.cvtColor(self.Image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_or(Image1, Image2)

        cv2.imshow("Image 1 Original", Image1)
        cv2.imshow("Image 2 Original", Image2)
        cv2.imshow("Image Operasi OR", operasi)

    def operasiXor(self):
        imagePath1, _ = QFileDialog.getOpenFileName()
        self.Image = cv2.imread(imagePath1)
        imagePath2, _ = QFileDialog.getOpenFileName()
        self.Image2 = cv2.imread(imagePath2)

        Image1 = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        Image2 = cv2.cvtColor(self.Image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_xor(Image1, Image2)

        cv2.imshow("Image 1 Original", Image1)
        cv2.imshow("Image 2 Original", Image2)
        cv2.imshow("Image Operasi XOR", operasi)

    def konv(self, X, F):
        X_height = X.shape[0]
        X_width = X.shape[1]

        F_height = F.shape[0]
        F_width = F.shape[1]

        H = (F_height) // 2
        W = (F_width) // 2

        out = np.zeros((X_height, X_width))

        for i in np.arange(H + 1, X_height - H):
            for j in np.arange(W + 1, X_width - W):
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum

        return out

    def konv2(self, X, F):
        X_Height = X.shape[0]
        X_Width = X.shape[1]

        F_Height = F.shape[0]
        F_Width = F.shape[1]

        H = 0
        W = 0

        batas = (F_Height) // 2

        out = np.zeros((X_Height, X_Width))

        for i in np.arange(H, X_Height - batas):
            for j in np.arange(W, X_Width - batas):
                sum = 0
                for k in np.arange(H, F_Height):
                    for l in np.arange(W, F_Width):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum
        return out

    def konvolusikernel1(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        img = self.Image
        id_kernel = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])
        hasil_konv = self.konv(self.Image, id_kernel)
        cv2.imshow('Original', self.Image)
        plt.imshow(hasil_konv, cmap='gray', interpolation='bicubic')
        plt.show()

    def konvolusikernel6(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        img = self.Image
        id_kernel = np.array([[6, 0, -6],
                              [6, 1, -6],
                              [6, 0, -6]])
        hasil_konv = self.konv(self.Image, id_kernel)
        cv2.imshow('Original', self.Image)
        plt.imshow(hasil_konv, cmap='gray', interpolation='bicubic')
        plt.show()

    def mean3x3(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        img = self.Image
        mean = (1.0 / 9) * np.array([[1/9, 1/9, 1/9],
                                     [1/9, 1/9, 1/9],
                                     [1/9, 1/9, 1/9]])
        hasil_mean = self.konv(self.Image, mean)
        cv2.imshow('Original', self.Image)
        plt.imshow(hasil_mean, cmap='gray', interpolation='bicubic')
        plt.show()

    def mean2x2(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        img = self.Image
        mean = (1.0 / 4) * np.array([[1/4, 1/4],
                                     [1/4, 1/4]])
        hasil_mean = self.konv2(self.Image, mean)
        cv2.imshow('Original', self.Image)
        plt.imshow(hasil_mean, cmap='gray', interpolation='bicubic')
        plt.show()

    def gauss(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        img = self.Image
        gauss = (1.0/345)* np.array([[1, 5, 7, 5, 1],
                                     [5, 20, 33, 20, 5],
                                     [7, 33, 55, 33, 7],
                                     [5, 20, 33, 20, 5],
                                     [1, 5, 7, 5, 1]])
        hasil_gauss = self.konv(self.Image, gauss)
        plt.imshow(hasil_gauss, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharpeningi(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        img = self.Image
        kerneli = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])

        hasil_sharp = self.konv(self.Image, kerneli)
        cv2.imshow('Original', self.Image)
        plt.imshow(hasil_sharp, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharpeningii(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        img = self.Image
        kernelii = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
        hasil_sharp = self.konv(self.Image, kernelii)
        cv2.imshow('Original', self.Image)
        plt.imshow(hasil_sharp, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharpeningiii(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        img = self.Image
        kerneliii = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
        hasil_sharp = self.konv(self.Image, kerneliii)
        cv2.imshow('Original', self.Image)
        plt.imshow(hasil_sharp, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharpeningiv(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        kerneliv = np.array([[1, -2, 1],
                             [-2, 5, -2],
                             [1, -2, 1]])
        hasil_sharp = self.konv(self.Image, kerneliv)
        cv2.imshow('Original', self.Image)
        plt.imshow(hasil_sharp, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharpeningv(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        img = self.Image
        kernelv = np.array([[1, -2, 1],
                            [-2, 4, -2],
                            [1, -2, 1]])
        hasil_sharp = self.konv(self.Image, kernelv)
        cv2.imshow('Original', self.Image)
        plt.imshow(hasil_sharp, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharpeningvi(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        img = self.Image
        kernelvi = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])
        hasil_sharp = self.konv(self.Image, kernelvi)
        cv2.imshow('Original', self.Image)
        plt.imshow(hasil_sharp, cmap='gray', interpolation='bicubic')
        plt.show()

    def sharpeninglaplace(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        img = self.Image
        laplace = (1.0/16)* np.array([[0, 0, -1, 0, 0],
                                      [0, -1, -2, -1, 0],
                                      [-1, -2, 16, -2, -1],
                                      [0, -1, -2, -1, 0],
                                      [0, 0, -1, 0, 0]])
        hasil_sharp = self.konv(self.Image, laplace)
        cv2.imshow('Original', self.Image)
        plt.imshow(hasil_sharp, cmap='gray', interpolation='bicubic')
        plt.show()

    def median(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        img = gray
        img_out = img.copy()
        h, w = img.shape[:2]

        for i in np.arange(3, h-3):
            for j in np.arange(3, w-3):
                neighbors= []
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i+k, j+1)
                        neighbors.append(a)
                neighbors.sort()
                median = neighbors[24]
                b = median
                img_out.itemset((i,j), b)
        med = img_out
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def maxFilter(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        img = gray
        img_out = img.copy()
        h, w = img.shape[:2]

        for i in np.arange(3, h-3):
            for j in np.arange(3, w-3):
                max = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i+k, j+1)
                        if a > max:
                            max = a
                b = max
                img_out.itemset((i,j), b)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def minFilter(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        img = gray
        img_out = img.copy()
        h, w = img.shape[:2]

        for i in np.arange(3, h-3):
            for j in np.arange(3, w-3):
                min = 255
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i+k, j+1)
                        if a < min:
                            min = a
                b = min
                img_out.itemset((i,j), b)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def DFT_Smooth(self):
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)

        y += max(y)

        img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)

        plt.imshow(img)
        img = cv2.imread("1.jpg", 0)

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 80
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 1

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')
        plt.show()

    def DFT_Edge(self):
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)

        y += max(y)

        img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)

        plt.imshow(img)
        img = cv2.imread("1.jpg", 0)

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 80
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')
        plt.show()

    def CannyEdge(self):
        # Noise Reduction
        img = self.Image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gauss = (1/57) * np.array([[0, 1, 2, 1, 0],
                                  [1, 3, 5, 3, 1],
                                  [2, 5, 9, 5, 2],
                                  [1, 3, 5, 3, 1],
                                  [0, 1, 2, 1, 0]])

        out_img = self.konv(img, gauss)
        out_img = out_img.astype("uint8")
        cv2.imshow("Noise Reduction", out_img)

        # Finding Gradien
        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

        img_x = self.konv(img, Sx)
        img_y = self.konv(img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out)) * 255
        img_out = img_out.astype("uint8")
        cv2.imshow("Finding Gradien", img_out)
        theta = np.arctan(img_y, img_x)

        # Non Maximum Supression
        angle = theta * 180 / np.pi
        angle[angle < 0] += 180
        H, W = img.shape[:2]
        Z = np.zeros((H, W), dtype=np.int32)
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255
                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img_out[i, j + 1]
                        r = img_out[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img_out[i + 1, j - 1]
                        r = img_out[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img_out[i + 1, j]
                        r = img_out[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img_out[i - 1, j - 1]
                        r = img_out[i + 1, j + 1]

                    if (img_out[i, j] >= q) and (img_out[i, j] >= r):
                        Z[i, j] = img_out[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        img_N = Z.astype("uint8")
        cv2.imshow("Non Maximum Supression", img_N)

        # Hysteresis Treshold p1
        weak = 100
        strong = 150
        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (a > weak):
                    b = weak
                    if (a > strong):
                        b = 255
                else:
                    b = 0
                img_N.itemset((i, j), b)
        img_H1 = img_N.astype("uint8")
        cv2.imshow("Hysteresis P1", img_H1)

        # Hysteresis Treshold p2
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if (img_H1[i, j] == weak):
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or
                                (img_H1[i + 1, j] == strong) or
                                (img_H1[i + 1, j + 1] == strong) or
                                (img_H1[i, j - 1] == strong) or (img_H1[i, j + 1] == strong) or
                                (img_H1[i - 1, j - 1] == strong) or
                                (img_H1[i - 1, j] == strong) or (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass
        img_H2 = img_H1.astype("uint8")
        cv2.imshow("Hysteresis P2", img_H2)

    def Countour(self):
        img = self.Image
        imgCanny = cv2.Canny(img, 10, 70)
        imgContours = img.copy()

        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
        self.Image = imgContours
        self.displayImage(2)

    def run(self):
        heightImg = 700
        widthImg = 700
        questions = 5
        choices = 5
        ans = [1, 2, 0, 2, 4]

        img = self.Image
        img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
        imgFinal = img.copy()
        imgBlank = np.zeros((heightImg, widthImg, 3),
                            np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
        imgCanny = cv2.Canny(imgBlur, 10, 70)  # APPLY CANNY

        try:
            ## FIND ALL COUNTOURS
            imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
            imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
            contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)  # FIND ALL CONTOURS
            cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS
            rectCon = utlis.rectContour(contours)  # FILTER FOR RECTANGLE CONTOURS
            biggestPoints = utlis.getCornerPoints(rectCon[0])  # GET CORNER POINTS OF THE BIGGEST RECTANGLE
            gradePoints = utlis.getCornerPoints(rectCon[1])  # GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE

            if biggestPoints.size != 0 and gradePoints.size != 0:

                # BIGGEST RECTANGLE WARPING
                biggestPoints = utlis.reorder(biggestPoints)  # REORDER FOR WARPING
                cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
                pts1 = np.float32(biggestPoints)  # PREPARE POINTS FOR WARP
                pts2 = np.float32(
                    [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
                matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GET TRANSFORMATION MATRIX
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))  # APPLY WARP PERSPECTIVE

                # SECOND BIGGEST RECTANGLE WARPING
                cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)  # DRAW THE BIGGEST CONTOUR
                gradePoints = utlis.reorder(gradePoints)  # REORDER FOR WARPING
                ptsG1 = np.float32(gradePoints)  # PREPARE POINTS FOR WARP
                ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # PREPARE POINTS FOR WARP
                matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)  # GET TRANSFORMATION MATRIX
                imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))  # APPLY WARP PERSPECTIVE`

                # APPLY THRESHOLD
                imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)  # CONVERT TO GRAYSCALE
                imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]  # APPLY THRESHOLD AND INVERSE

                boxes = utlis.splitBoxes(imgThresh)  # GET INDIVIDUAL BOXES
                countR = 0
                countC = 0
                myPixelVal = np.zeros((questions, choices))  # TO STORE THE NON ZERO VALUES OF EACH BOX
                for image in boxes:
                    # cv2.imshow(str(countR)+str(countC),image)
                    totalPixels = cv2.countNonZero(image)
                    myPixelVal[countR][countC] = totalPixels
                    countC += 1
                    if (countC == choices): countC = 0;countR += 1
                print(myPixelVal)

                # FIND THE USER ANSWERS AND PUT THEM IN A LIST
                myIndex = []
                for x in range(0, questions):
                    arr = myPixelVal[x]
                    myIndexVal = np.where(arr == np.amax(arr))
                    myIndex.append(myIndexVal[0][0])
                print("USER ANSWERS",myIndex)

                # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
                grading = []
                for x in range(0, questions):
                    if ans[x] == myIndex[x]:
                        grading.append(1)
                    else:
                        grading.append(0)
                print("GRADING",grading)
                score = (sum(grading) / questions) * 100  # FINAL GRADE
                print("SCORE",score)

                # DISPLAYING ANSWERS
                utlis.showAnswers(imgWarpColored, myIndex, grading, ans)  # DRAW DETECTED ANSWERS
                utlis.drawGrid(imgWarpColored)  # DRAW GRID
                imgRawDrawings = np.zeros_like(imgWarpColored)  # NEW BLANK IMAGE WITH WARP IMAGE SIZE
                utlis.showAnswers(imgRawDrawings, myIndex, grading, ans)  # DRAW ON NEW IMAGE
                invMatrix = cv2.getPerspectiveTransform(pts2, pts1)  # INVERSE TRANSFORMATION MATRIX
                imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))  # INV IMAGE WARP

                # DISPLAY GRADE
                imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)  # NEW BLANK IMAGE WITH GRADE AREA SIZE
                cv2.putText(imgRawGrade, str(int(score)) + "%", (70, 100)
                            , cv2.FONT_HERSHEY_COMPLEX, 3.5, (0, 255, 255), 7)  # ADD THE GRADE TO NEW IMAGE
                invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)  # INVERSE TRANSFORMATION MATRIX
                imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG,
                                                         (widthImg, heightImg))  # INV IMAGE WARP

                # SHOW ANSWERS AND GRADE ON FINAL IMAGE
                imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
                imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

                # IMAGE ARRAY FOR DISPLAY
                imageArray = ([img, imgGray, imgBlur, imgCanny, imgContours],
                              [imgBigContour, imgWarpGray, imgThresh, imgWarpColored, imgFinal])
        except:
            imageArray = ([img, imgGray, imgBlur, imgCanny, imgContours],
                          [imgBlank, imgBlank, imgBlank, imgBlank, imgBlank])

        # LABELS FOR DISPLAY
        lables = [["Original", "Gray", "Blur", "Edges", "Contours"],
                  ["Biggest Contour", "Warpped", "Threshold", "Mark", "Final"]]

        stackedImage = utlis.stackImages(imageArray, 0.5, lables)
        cv2.imshow("Result", stackedImage)
        cv2.waitKey()

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8
        if len(self.Image.shape)==3:
            if(self.Image.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0],
                     self.Image.strides[0], qformat)
        img = img.rgbSwapped()
        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)

        if windows == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Aplikasi Pendeteksi Lembar Jawaban')
window.show()
sys.exit(app.exec_())