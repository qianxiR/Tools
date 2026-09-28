import os,cv2
import numpy as np
from  skimage.io import imread, imsave
import PIL.Image
from tqdm import tqdm
PIL.Image.MAX_IMAGE_PIXELS=10000000000
from skimage.morphology import remove_small_objects, watershed
from skimage.morphology import erosion, dilation, rectangle,opening,closing,disk
from skimage.measure import label, regionprops
# [Warning: only fit to binary classified results]


#去除小图斑，针对不同类别，设定不同的阈值
"""
方案说明：
1.对于输入包含所有类的灰度图，将每一类转换成二值图的形式。然后根据指定图斑像素数量阈值，滤除像素数量小于阈值的图斑。
这些被滤除的图斑在二值图中的值就为0。
2.经过1操作的所有二值图进行逻辑或操作，这样就得到了除了背景和滤除的图斑处为0，其余位置均为1的二值图。
3.将背景也设置为1，那么被滤除的图斑处为0，其余地方均为1。
4.对3的结果取逻辑非，则被滤除的图斑处为1，其余地方均为0.
5.对4的结果进行区域处理。找出各个连通的图斑。找出图斑在原图中的区域，以及图斑包围矩形框上下左右各多一个像素的矩形框。
用这个矩形框内原图中的像素值减去图斑处的像素值，以及滤除掉背景像素值之后。将包围框内的非零元素进行重复频率统计，
选择频率最多的像素值作为图斑填充的像素值。
用此像素值填充图斑在原图中的区域。

八大类及二十五小类单独设置小图斑阈值,低于指定阈值像数个数以下的图斑直接滤除掉
八大类，二十五小类别：
1：耕地           2：园地            3: 林地           4: 草地
    11：水田        21：果园         31:有林地          41: 天然牧草地
    12：水浇地      22：茶园         32:灌木林地         42: 人工牧草地
    13：旱地        23：其他园地      33:其它林地        43: 其他草地  

5：建设用地          6：交通运输用地    7：水域及附属设施   8：其它
    51：城镇建设用地    61：农村道路       71：河湖库塘        81：盐碱地
    52：农村建设用地    62：其他交通用地    72：沼泽地         82：沙地
    53：采矿用地                        73：冰川及永久积雪   83：裸土地
    54：其他建设用地                                       84：裸岩石砾地

"""
def remove_spots(path,classes=8):
    im_path = path
    category_1 = {'1': 120, '2': 120, '3': 400, '4': 400, '5': 120, '6': 120, '7': 120, '8': 600}
    category_2 = {'11': 120, '12': 120, '13': 120, '21': 120, '22': 120, '23': 120,
                  '31': 400, '32': 400, '33': 400, '41': 400, '42': 400, '43': 400,
                  '51': 120, '52': 120, '53': 120, '54': 120, '61': 120, '62': 120,
                  '71': 120, '72': 120, '73': 120, '81': 600, '82': 600, '83': 600, '84': 600}

    # indicate which category standard to use,default "True" means category_1

    if classes == 8:
        category = category_1
    else:
        category = category_2
    im = cv2.imread(im_path)                   # 以灰度图的形式读入图像
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 转成灰度图
    assert im.any(), "File open failed. File path or file name may be wrong,please double check"
    num_cat = len(category.keys())
    h, w = im.shape
    out_im = np.zeros_like(im)
    im_copy = np.zeros((num_cat, h, w), dtype=bool)
    # 对每一类分别滤除指定阈值以下的小图斑，并对所有滤除了小图斑的二值图做逻辑或操作，这样得到的结果就是所有被滤除了的地方以及背景均为0，其他地方均为1
    for index, (key, value) in enumerate(category.items()):
        key = int(key)
        im_copy[index][im == key] = True
        rm_s = remove_small_objects(im_copy[index], min_size=value, connectivity=1, in_place=True)
        if index != 0:
            temp = np.logical_or(im_copy[index], im_copy[index-1])
            im_copy[index] = temp

    temp[im == 255] = 1               # 将背景像素全部设置为1。255为默认背景像素值，若背景像素值为其他数值，请将255修改为对应背景像数值。
    out_im[np.logical_not(temp)] = 1   # 这里将空洞0，也即被滤除掉的图斑位置像素值全部设为1，突出小图斑。
    labels = label(out_im, connectivity=1)  # 根据8连通的准则，将各个图斑进行编号
    region_properties = regionprops(labels)  # 各个独立连通图斑的属性对象，比如bbox，area，coords等。
    for region in region_properties:
        # 对每个独立的图斑区域，获取图斑的包围矩形，设置一个比包围矩形上下左右各多一个像素的大包围矩形区域。
        # 然后将大包围矩形区域在原图中的区域像素，减去图斑区域像素以及背景像素255，剩余的像素中，选择出现频率最多的像素值作为此图斑的填充像素值。
        top_x, top_y, bottom_x, bottom_y = region.bbox
        if top_x >= 1 and top_y >= 1 and bottom_x <= h-1 and bottom_y <= w-1:
            bbox_region = np.zeros((bottom_x-top_x+2, bottom_y-top_y+2))
            bbox_region[region.coords[:, 0]-top_x+1, region.coords[:, 1]-top_y+1] = im[region.coords[:, 0], region.coords[:, 1]]
            peri_bbox_region = im[top_x-1:bottom_x+1, top_y-1:bottom_y+1] - bbox_region
            peri_bbox_region[peri_bbox_region == 255] = 0
            peri_pxl = peri_bbox_region[peri_bbox_region.nonzero()].tolist()
            pixel_fill = int(max(set(peri_pxl), key=peri_pxl.count))
            im[region.coords[:, 0], region.coords[:, 1]] = pixel_fill
    # 将处理后的数据写入指定路径
    out_path = os.path.split(im_path)[0] + "\\" + "category_specific_remove_spots.png"
    cv2.imwrite(out_path, im)
    return out_path


def remove_small_objects_deal(predict, mask):
    lst = os.listdir(predict)
    for f in tqdm(lst):
        if not f.endswith(".png"):
            continue
        img_pre = imread(os.path.join(predict, f))
        # absname = os.path.split(f)[1]
        # img_mask = imread()
        # img_pre[np.where(img_pre==255)]=1
        # img_mask[np.where(img_mask==2)]=0
        # img_mask[ np.where ( img_mask == 6 ) ] = 1
        # img_mask[ np.where ( img_mask == 255 ) ] = 1
        # if img_mask.max() > 1:
        #     print("error")
        size = 29
        im_open = opening(img_pre, disk(size))
        img_pre = im_open
        img_pre = img_pre.astype(np.bool)
        img_pre = remove_small_objects(img_pre, 100).astype(np.uint8)

        imsave(os.path.join(mask, f), img_pre)


def dowork (predict,mask):
    lst = os.listdir(predict)
    for f in lst:
        if not f.endswith(".png"):
            continue
        img_pre = imread(os.path.join(predict, f))
        # absname = os.path.split(f)[1]
        # img_mask = imread()
        # img_pre[np.where(img_pre==255)]=1
        # img_mask[np.where(img_mask==2)]=0
        # img_mask[ np.where ( img_mask == 6 ) ] = 1
        # img_mask[ np.where ( img_mask == 255 ) ] = 1
        # if img_mask.max() > 1:
        #     print("error")
        size = 5
        im_open = opening(img_pre, disk(size))
        im_close = closing(im_open, disk(size))

        img_pre = im_open
        img_pre = img_pre.astype(np.bool)
        # img_pre = remove_small_objects(img_pre, 100).astype(np.uint8)
        imsave(os.path.join(mask, f), img_pre)


def calMetric(predict,mask):
    lst=os.listdir(predict)
    f_score=[]
    percision_score = [ ]
    recall_score = [ ]
    for f in lst:
        if not  f.endswith(".png"):
            continue
        img_pre=imread(os.path.join(predict,f))
        # absname = os.path.split(f)[1]
        img_mask=imread(os.path.join(mask,f))
        # img_pre[np.where(img_pre==255)]=1
        # img_mask[np.where(img_mask==2)]=0
        # img_mask[ np.where ( img_mask == 6 ) ] = 1
        # img_mask[ np.where ( img_mask == 255 ) ] = 1
        if img_mask.max()>1:
            print("error")
        size=3
        im_open = opening ( img_pre , disk ( size ) )
        img_pre = im_open
        img_pre = img_pre.astype(np.bool)
        img_pre = remove_small_objects(img_pre, 100).astype(np.uint8)
        im1=img_pre
        im2=img_mask
        valid = im1 >= 1
        iou = np.sum ( valid * (im1 == im2) )
        acc = np.sum ( (im1 == im2) ) / (im1.shape[ 0 ] * im1.shape[ 1 ])
        # iou=iou.sum()
        pre = im1.sum ( )
        p_true = im2.sum ( )
        f_score.append ( (2 * iou + 0.001) / (p_true + pre + 0.001) )
        percision_score.append(iou/pre)
        recall_score.append(iou/p_true)
        print ( '{} im1:{} p_true:{} iou:{} acc:{:.4f}  recall:{:.4f} precision:{:.4f}  f1:{:.4f}'.format ( f , pre ,
                                                                                                            p_true ,
                                                                                                            iou , acc ,
                                                                                                            iou / p_true ,
                                                                                                            iou / pre ,
                                                                                                            (
                                                                                                                        2 * iou + 0.001) / (
                                                                                                                        p_true + pre + 0.001) ) )


    print(f_score)

    print ("recall: "+ str(np.mean ( recall_score )) )
    print ( "percision: " + str ( np.mean ( percision_score ) ) )
    print ( "f: " + str ( np.mean ( f_score ) ) )


if __name__=="__main__":
    # remove_small_objects_deal("/home/omnisky/PycharmProjects/data/samples/global/test/pred/2019-12-30_10-06-38"
    #                           ,"/home/omnisky/PycharmProjects/data/samples/global/test/pred/post1")
    remove_small_objects_deal("/home/omnisky/PycharmProjects/data/rice/test/pred/classical/rice_all1_null_classical_unet_efficientnetb5_binary_crossentropy_adam_480_012bands_2020-03-14_16-55-24best"
              ,"/home/omnisky/PycharmProjects/data/rice/test/pred/classical/ts/")