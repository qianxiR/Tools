# -*-coding:utf-8-*-
import os
import sys
import subprocess
import shutil
import binascii
# import pyproj

grass7path = r'C:/Program Files/QGIS 3.16/apps/grass/grass78'   # 设置环境变量
grass7bin = r"C:/Program Files/QGIS 3.16/bin/grass78.bat"
# grass7bin = '"'+"C:/Program Files/QGIS 3.16/bin/grass78.bat"+'"'

def simplify_polygon(s_file_path,t_file_path,thd=1):
    #防止cmd中不识别带空格的路径，所以将grass7bin前后加上双引号
    print('简化面处理')
    startcmd = '"'+grass7bin +'"'+ ' --config path'
    print(startcmd)
    p = subprocess.Popen(startcmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    # Set GISBASE environment variable
    # print(pyproj.datadir.get_data_dir())
    gisbase = r"C:/Program Files/QGIS 3.16/apps/grass/grass78"
    os.environ['GISBASE'] = gisbase
    # define GRASS-Python environment
    gpydir = os.path.join(gisbase, "etc", "python")
    sys.path.append(gpydir)
    os.environ['PROJ_LIB'] = r'C:/Program Files/QGIS 3.16/share/proj'
    # define GRASS DATABASE
    gisdb = r'C:/data/grassdata'
    if not os.path.exists(gisdb):
        os.makedirs(gisdb)
    # location/mapset: use random names for batch jobs
    string_length = 16
    location = binascii.hexlify(os.urandom(string_length)).decode()
    mapset = 'PERMANENT'
    location_path = os.path.join(gisdb, location)
    # if not os.path.exists(location_path): #注释掉这两行，才能输出结果
    #     os.makedirs(location_path)
    #  from SHAPE or GeoTIFF file
    startcmd = '"'+grass7bin +'"' + ' -c ' + t_file_path + ' -e ' + location_path

    print(startcmd)
    # f=os.popen(startcmd,'r')

    p = subprocess.Popen(startcmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print(sys.stderr, 'ERROR: %s' % err)
        print(sys.stderr, 'ERROR: Cannot generate location (%s)' % startcmd)
        sys.exit(-1)
    else:
        print('Created location %s' % location_path)
    # Now the location with PERMANENT mapset exists.
    ########
    # Now we can use PyGRASS or GRASS Scripting library etc. after
    # having started the session with gsetup.init() etc

    import grass.script as grass
    import grass.script.setup as gsetup
    # import script as grass
    # import script.setup as gsetup

    ###########
    # Launch session and do something
    gsetup.init(gisbase, gisdb, location, mapset)
    grass.message('--- GRASS GIS 7: Current GRASS GIS 7 environment:')
    print(grass.gisenv())
    in_region = grass.region()
    grass.message("--- Computational region: '{}'".format(in_region))
    out_path = os.path.split(s_file_path)[0] + "\\simplify_polygon"
    filename = os.path.split(s_file_path)[-1]
    filename = filename.split('.')[0]
    # generalize 方法可选
    grass.run_command('v.in.ogr', input=s_file_path, output='shapefile')
    # grass.run_command('v.import', input=file_path, output='shapefile')
    # grass.run_command('v.generalize', input='shapefile', output='simplify_polygon', method='douglas', threshold=thd)
    # grass.run_command('v.out.ogr', input='simplify_polygon', output=out_path, format="ESRI_Shapefile", overwrite=True)
    grass.run_command('v.generalize', input='shapefile', output=filename, method='douglas', threshold=thd)
    grass.run_command('v.out.ogr', input=filename, output=out_path, format="ESRI_Shapefile", overwrite=True)

    # Finally remove the temporary batch location from disk
    print('Removing location{}'.format(location_path))
    shutil.rmtree(location_path)
    sms_file_path = os.path.join(out_path,os.path.split(s_file_path)[-1])
    return sms_file_path


if __name__ == "__main__":
    # shp_file = r'C:/data/changedetect/tfxq/result/2022-04-18_22-14-45/test_1.shp'
    # tif_file = r'C:/data/changedetect/tfxq/result/2022-04-18_22-14-45/test_1.tif'

    shp_file = r'C:/data/chuangshudata/result/2022-05-09_22-53-56/qbj_proj_Clip1.shp'
    tif_file = r'C:/data/chuangshudata/result/2022-05-09_22-53-56/qbj_proj_Clip1.tif'
    simplify_polygon(shp_file, tif_file, thd=3)
