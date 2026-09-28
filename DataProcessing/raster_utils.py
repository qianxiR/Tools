# -*- coding: utf-8 -*-
"""
栅格 I/O 基础工具
GDAL 影像加载、波段提取、文件检索、坐标匹配、多边形化等底层功能。

入参/方法/出参 见各函数注释。
"""

import os
import sys
import time
import cv2
import gc
import numpy as np
from osgeo import gdal, osr, ogr
gdal.UseExceptions()

UINT8 = 0
UINT10 = 1
UINT16 = 2


class FError(Exception):
    pass


ERROR_OPEN_IMAGEFILE = -110
ERROR_NUMPY_TRANSPOSE = -220


class Base_ulitities:
    def __init__(self):
        return 0

    def load_img_by_cv2(self, path, grayscale=False):
        """
        :param path: input image file path
        :param grayscale:  bool value
        :return: flag, image values
        """
        if not os.path.isfile(path):
            print("input path is not a file!")
            return -1, None
        if grayscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)
        return 0, img


def geotrans_match(infile, outfile, info=[]):
    try:
        indataset = gdal.Open(infile)
    except:
        print("Waring: cannot open input file:{}".format(infile))
        return -1
    in_geotrans = indataset.GetGeoTransform()
    info.append(in_geotrans)
    try:
        in_geoproj = indataset.GetProjection()
    except:
        print('Warning:获取投影信息失败')

    del indataset

    outdataset = gdal.Open(outfile, gdal.GA_Update)
    outdataset.SetGeoTransform(in_geotrans)
    try:
        outdataset.SetProjection(in_geoproj)
    except:
        print("no projection info transfer to output file:{}".format(outfile))

    del outdataset
    return 0


def echoRuntime(func):
    def wrapper(*args, **kwargs):
        startTime = time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())
        t1 = time.time()
        print(func.__name__ + ">>>......Strat at : " + str(startTime))
        func(*args, **kwargs)
        endTime = time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())
        t2 = time.time()
        print(func.__name__ + ">>>......Finished at : " + endTime)
        msecs = (t2 - t1)
        print(func.__name__ + " running time is %.2f s" % msecs)
    return wrapper


def send_message_callback(text):
    print(text)


def load_img_by_cv2(path, grayscale=False):
    """
    :param path: input image file path
    :param grayscale:  bool value
    :return: flag, image values
    """
    if not os.path.isfile(path):
        print("input is not a file!")
        return -1, None
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
    return 0, img


def load_img_normalization_by_cv2(path, grayscale=False):
    if not os.path.isfile(path):
        print("input path is not a file!")
        return -1, None
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0
    return 0, img


def load_label(path):
    if not os.path.isfile(path):
        print("input path is not a file!")
        return -1
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    except:
        print("Openv loading label failed, try gdal")
        try:
            dataset = gdal.Open(path)
        except:
            print("gdal loading label also failed")
            return -2
        img = dataset.ReadAsArray()
        del dataset
    if img is None:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 0)
        if img is None:
            print("read label image failed")
            return -2
    return img


def load_src(path, bandlist=[], mode=0):
    """
    :param path: sample direcotry containing two dir label and src
    :param bandlist: a list such as [0,1,2]
    :param mode: src image load mode, 0-uint8, 1-uint16
    :return:
    """
    if not os.path.isfile(path):
        print("no file:{}".format(path))
        return -1
    dataset = gdal.Open(path)
    if dataset is None:
        print("Open file failed:{}".format(path))
        return -2
    y_height = dataset.RasterYSize
    x_width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    im_type = dataset.GetRasterBand(1).DataType
    img = dataset.ReadAsArray(0, 0, x_width, y_height)
    if len(bandlist) == 0:
        bandlist = range(im_bands)
    if len(bandlist) > im_bands or max(bandlist) >= im_bands:
        print("input bands should not be bigger than image bands!")
        return -3

    """transpose format to: band last"""
    img = np.asarray(img, np.float16)
    a, _, _ = img.shape
    if im_bands == 1:
        img = np.expand_dims(img, axis=-1)
    else:
        if im_bands == a:
            img = np.transpose(img, (1, 2, 0))
    if mode == 0:
        out_img = np.zeros((y_height, x_width, len(bandlist)), np.uint8)
    elif mode == 1:
        out_img = np.zeros((y_height, x_width, len(bandlist)), np.uint16)
    else:
        print("src mode should be 0 or 1")
        return -4
    try:
        for i in range(len(bandlist)):
            out_img[:, :, i] = img[:, :, bandlist[i]]
    except:
        print("Error: could not get correct list of data from image")
        return -4
    else:
        pass

    del dataset
    return out_img


def load_img_by_gdal(path, grayscale=False):
    try:
        dataset = gdal.Open(path)
    except RuntimeError:
        print("Could not open file:{}".format(path))
        return []

    y_height = dataset.RasterYSize
    x_width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    img = dataset.ReadAsArray(0, 0, x_width, y_height)
    if grayscale == False and im_bands > 1:
        img = np.asarray(img, np.uint16)
        try:
            img = np.transpose(img, (1, 2, 0))
        except:
            print("image should be 3 dimensions!{}".format(path))
            return []
    else:
        if im_bands > 1:
            img = np.transpose(img, (1, 2, 0))
    del dataset

    return img


def load_img_by_gdal_new(path, img=[], grayscale=False):
    try:
        dataset = gdal.Open(path)
    except RuntimeError:
        print("Could not open file:{}".format(path))
        return ERROR_OPEN_IMAGEFILE

    y_height = dataset.RasterYSize
    x_width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    img = dataset.ReadAsArray(0, 0, x_width, y_height)

    if grayscale == False:
        img = np.array(img, dtype="float")
        try:
            img = np.transpose(img, (1, 2, 0))
        except:
            print("image should be 3 dimensions!")
            return ERROR_NUMPY_TRANSPOSE
    else:
        if im_bands > 1:
            img = np.transpose(img, (1, 2, 0))
    del dataset

    return 0


def load_img_by_gdal_geo(path, grayscale=False):
    try:
        dataset = gdal.Open(path)
    except RuntimeError:
        return [], ''
    y_height = dataset.RasterYSize
    x_width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    img = dataset.ReadAsArray(0, 0, x_width, y_height)
    geotransform = dataset.GetGeoTransform()

    if grayscale == False:
        img = np.array(img, dtype="float")
        try:
            img = np.transpose(img, (1, 2, 0))
        except:
            print("image should be 3 dimensions!")
            return [], ''
    else:
        if im_bands > 1:
            img = np.transpose(img, (1, 2, 0))
    del dataset

    return img, geotransform


def load_img_by_gdal_info(path, grayscale=False):
    try:
        dataset = gdal.Open(path)
    except RuntimeError:
        return 0, 0, 0, '', ''

    y_height = dataset.RasterYSize
    x_width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    geotransform = dataset.GetGeoTransform()
    projectInfo = dataset.GetProjectionRef()

    del dataset

    return y_height, x_width, im_bands, geotransform, projectInfo


def load_img_by_gdal_blocks(path, x, y, width, height, grayscale=False):

    dataset = gdal.Open(path)

    y_height = dataset.RasterYSize
    x_width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    if y + height > y_height:
        height = y_height - y
    img = dataset.ReadAsArray(x, y, width, height)
    if grayscale == False:
        img = np.asarray(img, np.uint16)
        try:
            img = np.transpose(img, (1, 2, 0))
        except:
            print("image should be 3 dimensions!")
            sys.exit(-1)
    else:
        if im_bands > 1:
            img = np.transpose(img, (1, 2, 0))
    del dataset
    gc.collect()

    return img


def load_img_normalization(input_bands, path, data_type=UINT8):
    if not os.path.isfile(path):
        print("input path is not a file!")
        return -1, None
    if input_bands == 1 and data_type == UINT8:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 0)
    elif input_bands == 3 and data_type == UINT8:
        img = cv2.imread(path)
        if img is None:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint16), 1)
        img = np.array(img, dtype="float") / 255.0
    else:
        img = load_img_by_gdal(path)
        if data_type == UINT8:
            img = img / 255.0
        elif data_type == UINT10:
            img = img / 1024.0
        elif data_type == UINT16:
            img = img / 25535.0
        else:
            print("Not recognize this type!")
            sys.exit(-1)
        img = np.clip(img, 0.0, 1.0)
    return 0, img


def load_img_normalization_bybandlist(path, bandlist=[], data_type=UINT8):
    dataset = gdal.Open(path)

    y_height = dataset.RasterYSize
    x_width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    img = dataset.ReadAsArray(0, 0, x_width, y_height)
    if len(bandlist) == 0:
        bandlist = range(im_bands)
    if len(bandlist) > im_bands or max(bandlist) >= im_bands:
        print("input bands should not be bigger than image bands!")
        sys.exit(-2)

    """transpose format to: band last"""
    img = np.array(img, dtype="float")
    if im_bands == 1:
        img = np.expand_dims(img, axis=-1)
    else:
        img = np.transpose(img, (1, 2, 0))

    out_img = np.zeros((y_height, x_width, len(bandlist)), np.float16)
    try:
        for i in range(len(bandlist)):
            out_img[:, :, i] = img[:, :, bandlist[i]]
    except:
        print("Error: could not get correct list of data from image")
        sys.exit(-5)
    else:
        pass

    if data_type == UINT8:
        out_img = out_img / 255.0
    elif data_type == UINT10:
        out_img = out_img / 1024.0
    elif data_type == UINT16:
        out_img = out_img / 25535.0
    else:
        print("Not recognize this type!")
        sys.exit(-1)
    out_img = np.clip(out_img, 0.0, 1.0)

    del dataset
    return 0, out_img


def load_img_bybandlist(path, bandlist=[]):
    if not os.path.isfile(path):
        print("no file:{}".format(path))
        return -1, []
    dataset = gdal.Open(path)
    if dataset is None:
        print("Open file failed:{}".format(path))
        return -1, []
    y_height = dataset.RasterYSize
    x_width = dataset.RasterXSize
    im_bands = dataset.RasterCount
    img = dataset.ReadAsArray(0, 0, x_width, y_height)
    if len(bandlist) == 0:
        bandlist = range(im_bands)
    if len(bandlist) > im_bands or max(bandlist) >= im_bands:
        print("input bands should not be bigger than image bands!")
        return -1, img

    """transpose format to: band last"""
    img = np.array(img, dtype="float")
    a, _, _ = img.shape
    if im_bands == 1:
        img = np.expand_dims(img, axis=-1)
    else:
        if im_bands == a:
            img = np.transpose(img, (1, 2, 0))

    out_img = np.zeros((y_height, x_width, len(bandlist)), np.uint16)
    try:
        for i in range(len(bandlist)):
            out_img[:, :, i] = img[:, :, bandlist[i]]
    except:
        print("Error: could not get correct list of data from image")
        return -2, img
    else:
        pass

    del dataset
    return 0, out_img


def get_file(file_dir, file_type=[".jpg", '.png', '.tif', '.tiff', '.img', '.pix', '.bmp']):
    """
    :param file_dir: directory of input files, it may have sub_folders
    :param file_type: file format, namely postfix, a str or list
    :return: L: a list of files under the file_dir and sub_folders; num: the length of L
    """
    im_type = ['.png']
    if isinstance(file_type, str):
        im_type = file_type
    elif isinstance(file_type, list):
        im_type = file_type
    L = []
    if not os.path.isdir(file_dir):
        print("Error:input dir is not existed")
        return "", 0
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if (str.lower(os.path.splitext(file)[1]) in im_type):
                L.append(os.path.join(root, file))
    num = len(L)
    if num == 0:
        L = ""
    return L, num


def get_file_absname(file_dir, file_type=['.png', '.PNG', '.tif', '.img', '.IMG']):
    """
    :param file_dir: directory of input files, it may have sub_folders
    :param file_type: file format, namely postfix
    :return: L: a list of files under the file_dir and sub_folders; num: the length of L
    """
    im_type = ['.png']
    if isinstance(file_type, str):
        im_type = file_type
    elif isinstance(file_type, list):
        im_type = file_type
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if (os.path.splitext(file)[1] in im_type):
                absname = os.path.split(file)[1]
                L.append(absname)
    return L


def find_file(dir, str):
    """
    find the file containing str in dir
        :param dir: directory , it may have sub_folders
        :param str: string
        :return: file: a file with name of str in dir
    """
    temp = ''
    files, nb = get_file(dir)
    for file in files:
        if not str in file:
            continue
        else:
            temp = file
    else:
        return temp


""" check the size of src_img and label_img"""
def compare_two_image_size(img_one, img_two, grayscale=False):
    if grayscale:
        h1, w1 = img_one.shape
        h2, w2 = img_two.shape
        assert (h1 == h2 and w1 == w2)
    else:
        h1, w1, _ = img_one.shape
        h2, w2, _ = img_two.shape
        assert (h1 == h2 and w1 == w2)


def polygonize(rasterTemp, outShp, sieveSize=1):
    sourceRaster = gdal.Open(rasterTemp)
    band = sourceRaster.GetRasterBand(1)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outShp):
        driver.DeleteDataSource(outShp)

    outDatasource = driver.CreateDataSource(outShp)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(sourceRaster.GetProjectionRef())
    outLayer = outDatasource.CreateLayer(outShp, srs)

    newField = ogr.FieldDefn('grid_code', ogr.OFTInteger)
    outLayer.CreateField(newField)

    gdal.Polygonize(band, None, outLayer, 0, [], callback=None)

    outDatasource.Destroy()
    sourceRaster = None
    band = None

    del sourceRaster, band, outDatasource
    gc.collect()

    return 0


if __name__ == "__main__":

    rasterfile = "C:/data/changedetect/tfxq/result/2022-04-18_22-14-45/test_1.tif"
    shpfile = "C:/data/changedetect/tfxq/result/2022-04-18_22-14-45/test_1.shp"
    ret = 0
    ret = polygonize(rasterfile, shpfile)
    print(ret)
