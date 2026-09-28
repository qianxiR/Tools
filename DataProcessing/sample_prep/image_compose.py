#coding:utf-8
import os,sys,fire
from Tools.DataProcessing.raster_utils import get_file,find_file,geotrans_match
from tqdm import tqdm
import numpy as np
try:
    from osgeo import ogr, osr, gdal
    gdal.UseExceptions()
except:
    sys.exit('ERROR: cannot find GDAL/OGR modules')


def img_compose_blocks(file_list,output,BType=0,block_h=10000):
    result = []
    try:
        dataset_a = gdal.Open(file_list[0])
    except:
        print("Warning: file opening failed :\n {}".format(file_list[0]))
        return -1
    try:
        dataset_b = gdal.Open(file_list[1])
    except:
        print("Warning: file opening failed :\n {}".format(file_list[1]))
        return -2

    # try:
    #     prj = dataset_a.GetProjectionRef()
    #     geotransform = dataset_a.GetGeoTransform()
    # except:
    #     prj =''
    #     print("warning: get projection ref failed")



    band_n_a = dataset_a.RasterCount
    band_n_b = dataset_b.RasterCount
    band_n = band_n_a + band_n_b

    width = dataset_a.RasterXSize
    height = dataset_a.RasterYSize
    if width != dataset_b.RasterXSize or height != dataset_b.RasterYSize:
        print("Warning:input images must have the same width and height \n {}".format(os.path.split(file_list[0])[1]))
        return -3

    bk_h = block_h
    if block_h > height:
        print("block height is gt image height")
        bk_h = height
        # return -2

    try:
        driver = gdal.GetDriverByName("GTiff")
        if BType==0:
            outdataset = driver.Create(output, width, height, band_n, gdal.GDT_Byte)
        elif BType==1:
            outdataset = driver.Create(output, width, height, band_n, gdal.GDT_UInt16)
        else:
            print("Warning: only support 8bits and 16bits\n")
            return -4
    except:
        print("Error: output raster is existed or can not be opened:\n {}".format(output))
        return -3

    # if len(prj)>1:
    #     outdataset.SetProjection(prj)
    #     outdataset.SetGeoTransform(geotransform)


    n_blocks = 1
    if height % bk_h == 0:
        n_blocks = int(height / bk_h)
    else:
        n_blocks = int(height / bk_h) + 1

    # real_b = min(im_bands, bands)
    for index_block in range(n_blocks):
        start_y = index_block * bk_h
        y_block_height = bk_h
        if index_block == n_blocks - 1:
            y_block_height = height - start_y

        for i in range(band_n_a):
            tmp_band = dataset_a.GetRasterBand(i + 1)
            tmp = tmp_band.ReadAsArray(0, start_y, width, y_block_height)
            outdataset.GetRasterBand(i + 1).WriteArray(tmp, xoff=0, yoff=start_y)

        for i in range(band_n_b):
            tmp_band = dataset_b.GetRasterBand(i + 1)
            tmp = tmp_band.ReadAsArray(0, start_y, width, y_block_height)
            outdataset.GetRasterBand(band_n_a + i + 1).WriteArray(tmp, xoff=0, yoff=start_y)

    outdataset.FlushCache()
    del dataset_b, dataset_a, outdataset

    geotrans_match(file_list[0], output)
    return 0



def batch_image_compose(dir_pre,dir_post,outpath,BitType=0,bk_h=20000):
    if not os.path.isdir(dir_pre) or not os.path.isdir(dir_post):
        print("Error:input is not a directory")
        return -1
    filelist_a, nb = get_file(dir_pre)
    if nb == 0:
        print("Error: there is no file in dir a")
        return -3
    if not os.path.isdir(outpath):
        print("Warning: outdir is not exist, it will be created")
        os.mkdir(outpath)

    for fileA in tqdm(filelist_a):
        basename = os.path.basename(fileA).split(".")[0]
        fileB = find_file(dir_post, basename)
        if len(fileB) == 0:
            print("Warning: corresponding index file is not exist \n {}".format(os.path.split(fileA)[1]))
            continue
        print("前期影像：{}".format(fileA))
        print("后期影像：{}".format(fileB))
        flist = []
        flist.append(fileA)
        flist.append(fileB)
        outfile = outpath + '/' + basename + '.tif'
        if os.path.isfile(outfile):
            print("Warning:result file is existed:{}\n".format(outfile))
            continue
        ret = 0
        ret = img_compose_blocks(flist, outfile, block_h=bk_h)
        if ret != 0:
            print("Error:combinig failed :{}".format(basename))
            continue

    return 0






if __name__=="__main__":
    fire.Fire()
    print("")