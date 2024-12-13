import os  # 文件处理库
import tarfile  # 压缩包处理库
import numpy as np  # 科学计算库
from osgeo import gdal, osr  # 处处理栅格数据的库
import h5py
import cv2  # 图像处理的库
import readers as rd  # 读取光子的函数
import matplotlib.pyplot as plt  # 绘图库
import datetime
from sklearn.cluster import DBSCAN  # DBSCAN聚类
from sklearn.preprocessing import StandardScaler  # 数据预处理的库
from tqdm import *
from scipy import spatial
from scipy import io


def decompress_package(path):
    """
    用于解压Landsat的压缩包
    :param path:压缩包的路径
    :return:返回解压后的文件的位置
    """
    compress_dir = os.path.split(path)[0]  # 压缩文件的路径
    compress_name = os.path.split(path)[1]  # 压缩文件的文件名
    decompress_dir = compress_dir + f"/{os.path.splitext(compress_name)[0]}/"  # 合成解压的文件夹
    if not os.path.exists(decompress_dir):  # 创建文件夹
        os.makedirs(decompress_dir)
    try:
        tar = tarfile.open(path)
        for tarinfo in tar:
            if os.path.splitext(tarinfo.name)[1] == ".TIF" and os.path.splitext(tarinfo.name)[0][-2:-1] == "B":
                tar.extract(tarinfo.name, decompress_dir)  # 解压在文件夹中
        tar.close()  # 关闭压缩包
    except Exception:
        raise Exception
    return decompress_dir


def delete_decompress_package(path):
    """
    用于删除解压后的临时文件
    :param path: 解压后的临时文件所在路径
    :return:
    """
    files = os.listdir(path)  # 得到文件夹下的所有文件
    for file in files:  # 遍历文件夹下的所有文件
        os.remove(path + file)  # 拼接为完整路径，然后删除
    os.rmdir(path)  # 删除文件夹


def export_band(path, band):
    """
    读取特定波段数据，返回波段矩阵数组
    :param path:波段文件所在路径
    :param band:波段 ["B1","B2",.....]
    :return:返回波段图像数组
    """

    if band == False:
        ds = gdal.Open(path)
        data = ds.GetRasterBand(1).ReadAsArray()
        return data
    else:
        files = os.listdir(path)
        for file in files:
            if os.path.splitext(file)[1] == ".TIF" and os.path.splitext(file)[0][-2:] == band:
                ds = gdal.Open(path + file)
                data = ds.GetRasterBand(1).ReadAsArray()
                return data


def lontat_to_xy(lon, lat, gcs, pcs):
    """
    经纬度坐标转换为投影坐标
    :param lon:
    :param lat:
    :return:
    """
    # 经纬度坐标转换为投影坐标
    ct = osr.CoordinateTransformation(gcs, pcs)
    coordinates = ct.TransformPoint(lat, lon)
    return coordinates[0], coordinates[1], coordinates[2]


def xy_to_rowcol(x, y, GeoTransform):
    # 投影坐标转为栅格的栅格坐标
    a = np.array([[GeoTransform[1], GeoTransform[2]], [GeoTransform[4], GeoTransform[5]]])
    b = np.array([x - GeoTransform[0], y - GeoTransform[3]])
    row_col = np.linalg.solve(a, b)
    row = int(np.floor(row_col[1]))
    col = int(np.floor(row_col[0]))
    return row, col


def cropped_band(band, lat_lon_info, GP_information, pg_information):
    """
    裁剪波段影像，根据经纬度[[左上角经度，左上角纬度],[右下角经度，右下角纬度]]来裁剪波段
    :param band: 波段数据
    :param lan_lon_info: [[左上角经度，左上角纬度],[右下角经度，右下角纬度]]
    :param GP_information: 投影转换信息，投影信息
    :param pg_information: 坐标转换器
    :return:返回裁剪后的波段影像
    """
    row_col = []
    shape_x, shape_y = np.shape(band)  # 得到行数和列数
    for lonlat in lat_lon_info:
        x, y, _ = lontat_to_xy(lonlat[1], lonlat[0], pg_information[0], pg_information[1])  # 投影坐标
        row, col = xy_to_rowcol(x, y, GP_information[0])  # 栅格坐标
        # 若不在图像范围内，也进行裁剪
        if row < 0:
            row = 0
        if col < 0:
            col = 0
        if row > shape_x:
            row = shape_x
        if col > shape_y:
            col = shape_y
        row_col.append([row, col])
    band_crop = band[row_col[0][0]:row_col[1][0], row_col[0][1]:row_col[1][1]]
    return band_crop


def save_tiff(band, grid_info, path):
    """
    保存提取出来的水体，保存为整数形式，0与255的二值化栅格图像
    :param band: 水体指数的图像数组
    :param grid_info: 栅格图像的地理信息 [仿射变换参数，投影信息]
    :param path: 保存的tif文件名"./path/name.tif"
    :return:
    """
    x, y = band.shape  # 得到二维数组的行数
    diver = gdal.GetDriverByName('GTiff')  # 为注册哪种格式的数据。通常影像为Geotiff格式，载入数据驱动，初始化一个对象
    new_dataset = diver.Create(path, y, x, 1, gdal.GDT_Byte)  # float双精度保存，以保留小数点
    new_dataset.SetGeoTransform(grid_info[0])  # 写入仿射变换参数
    new_dataset.SetProjection(grid_info[1])  # 写入投影信息
    new_dataset.GetRasterBand(1).WriteArray(band)
    del new_dataset  # 删除空间


def get_tif_information(path):
    """
    得到一个Landsat压缩包里的tif文件的地理信息, 或者一个tif文件的文件信息
    :param path:解压后的文件路径
    :return: 返回[仿射变换参数，投影信息] ,坐标转换相关信息
    """
    if os.path.isdir(path):  # 判断路径是否问文件夹
        files = os.listdir(path)
        for file in files:
            if os.path.splitext(file)[1] == ".TIF" and os.path.splitext(file)[0][-2:] == "B1":
                ds = gdal.Open(path + file)
                GeoTransform = ds.GetGeoTransform()  # 投影转换信息（仿射坐标变化的信息，即栅格坐标和地理坐标变换的信息），地理仿射变换参数 [左上角x坐标，水平空间分辨率，行旋转，左上角y坐标，列旋转，垂直空间分辨率]
                ProjectionInfo = ds.GetProjection()  # 栅格数据的投影，获得投影信息

                pcs = osr.SpatialReference()  # 用来描绘坐标系统（投影和基准面）
                pcs.ImportFromWkt(ProjectionInfo)  # 将WKT坐标系统设置到OGRSpatialReference中
                gcs = pcs.CloneGeogCS()  # 读取投影的地理基准，用于和地理坐标进行转换

                return [GeoTransform, ProjectionInfo], [gcs, pcs]

    else:
        ds = gdal.Open(path)
        GeoTransform = ds.GetGeoTransform()  # 投影转换信息（仿射坐标变化的信息，即栅格坐标和地理坐标变换的信息），地理仿射变换参数 [左上角x坐标，水平空间分辨率，行旋转，左上角y坐标，列旋转，垂直空间分辨率]
        ProjectionInfo = ds.GetProjection()  # 栅格数据的投影，获得投影信息

        pcs = osr.SpatialReference()  # 用来描绘坐标系统（投影和基准面）
        pcs.ImportFromWkt(ProjectionInfo)  # 将WKT坐标系统设置到OGRSpatialReference中
        gcs = pcs.CloneGeogCS()  # 读取投影的地理基准，用于和地理坐标进行转换

        return [GeoTransform, ProjectionInfo], [gcs, pcs]


def crop_NDWI(path, lat_lon_info):
    """
    得到经纬度范围内的NDWI值，并保存为tif
    :param path: 遥感影像压缩包文件所在路径
    :param lat_lon_info: 经纬度范围
    :return:
    """
    decompress_dir = decompress_package(path)
    date = path.split(".tar")[-2].split("_")[-4]
    GP_information, pg_information = get_tif_information(decompress_dir)

    B5 = export_band(decompress_dir, "B5")  # 近红外波段
    B3 = export_band(decompress_dir, "B3")  # 绿色波段

    B5 = cropped_band(B5, lat_lon_info, GP_information, pg_information)
    B3 = cropped_band(B3, lat_lon_info, GP_information, pg_information)

    B5 = np.ma.masked_equal(B5, 0)
    B3 = np.ma.masked_equal(B3, 0)

    NDWI = (B3 - B5) / (B3 + B5)
    water_NDWI = np.where((0 <= NDWI) & (NDWI <= 0.3), 0, 255)

    save_tiff(water_NDWI, GP_information, f"./121040_water_body/{date}_NDWI.tif")

    delete_decompress_package(decompress_dir)


def binary_image_stretching(data, i):
    """
    用于二值图像拉伸，取三倍标准偏差，增强对比度
    :param data:二值化图像的二维数据
    :return:返回截止处的最大值和最小值，最大值不大于数据的最大值，最小值不小于数据的最小值
    """
    mean = np.nanmean(data)
    std_range = np.nanstd(data)
    vmin = max(mean - i * std_range, np.nanmin(data))
    vmax = min(mean + i * std_range, np.nanmax(data))

    data = np.where((data >= vmin) & (data <= vmax), data, np.nan)
    data = data[~np.isnan(data)]

    return vmin, vmax, data


def DBSCAN_denosing(ATDS, h_ph, num1, num2):
    """传入沿轨距离和高程，对数据进行预处理后
       进行DBSCAN去噪，然后返回噪声点和信号点的分类标签
       -1为噪声光子，0为信号光子"""
    """ATDS为沿轨距离，h_ph为高程"""
    """num1为半径Eps参数大小"""
    """num2为MinPts最小样本数的大小"""

    """数据预处理"""
    X = []
    for index in range(len(ATDS)):
        X.append([ATDS[index], h_ph[index]])
    X = np.asarray(X)
    X = StandardScaler().fit_transform(X)  # 数据预处理，进行标准化

    """DBSCAN去噪"""
    db = DBSCAN(eps=num1, min_samples=num2).fit(X)  # 进行DBSCAN去噪
    DBSCAN_labels = db.labels_  # 得到DBSCAN分类标签

    """提取信号光子和噪声光子"""
    DBSCAN_labels = np.where(DBSCAN_labels == -1, False, True)

    """
    unique_values, occurrence_count = np.unique(DBSCAN_labels, return_counts=True)  # 获取一维数组唯一值以及唯一值的频数
    num_index = np.argsort(occurrence_count)  # 获得升序排序后的索引值
    max_labels1 = unique_values[num_index[-1]]  # 获取频率最大的标签                                #获取频率最大的标签
    DBSCAN_labels[DBSCAN_labels == max_labels1] = 0  # 除频率最大的组，其他组作为噪声
    DBSCAN_labels[DBSCAN_labels != 0] = -1
    """
    """返回噪声和信号光子标签"""
    return np.asarray(DBSCAN_labels)


def one_histogram_denoising(df03_xs, df03_heights):
    """
        一次粗去噪，降低数据量
        :param df03_xs:
        :param df03_heights:
        :return:
    """
    print("one_histogram_denoising")
    plt.figure(0)

    bin_counts = int(np.sqrt(len(df03_heights)))
    counts, bin_wideth, _ = plt.hist(df03_heights, bin_counts, )  # 画出直方图

    # 得到频数最大的bin，得到最大bin前后范一个的bin,将此三个直方图内的高程继续做一次直方图
    max_counts = np.max(counts)
    max_index = np.where(counts == max_counts)[0][0] + 1
    max_height = np.max(bin_wideth[max_index - 1:max_index + 1])
    min_height = np.min(bin_wideth[max_index - 1:max_index + 1])

    df03_xs = np.where((df03_heights <= max_height) & (df03_heights >= min_height), df03_xs, np.nan)
    df03_heights = np.where((df03_heights <= max_height) & (df03_heights >= min_height), df03_heights, np.nan)

    df03_xs = df03_xs[~np.isnan(df03_xs)]
    df03_heights = df03_heights[~np.isnan(df03_heights)]

    plt.close()

    return df03_xs, df03_heights


def twice_histogram_denoising_and_outlier_removal(df03_xs, df03_heights):
    """
    选取两次直方图最大频数的bin和前后一个bin的范围高程，然后使用一倍方差去除异常值得到信号光子，提取高程
    :param df03_xs:
    :param df03_heights:
    :return:
    """
    print('twice_histogram_denoising_and_outlier_removal')
    plt.figure(0)

    bin_counts = int(np.sqrt(len(df03_heights)))
    counts, bin_wideth, _ = plt.hist(df03_heights, bin_counts, )  # 画出直方图

    # 得到频数最大的bin，得到最大bin前后范一个的bin,将此三个直方图内的高程继续做一次直方图
    max_counts = np.max(counts)
    max_index = np.where(counts == max_counts)[0][0] + 1
    max_height = bin_wideth[max_index + 1]
    min_height = bin_wideth[max_index - 2]

    df03_xs = np.where((df03_heights <= max_height) & (df03_heights >= min_height), df03_xs, np.nan)
    df03_heights = np.where((df03_heights <= max_height) & (df03_heights >= min_height), df03_heights, np.nan)

    df03_xs = df03_xs[~np.isnan(df03_xs)]
    df03_heights = df03_heights[~np.isnan(df03_heights)]

    bin_counts = int(np.sqrt(len(df03_heights)))
    counts, bin_wideth, _ = plt.hist(df03_heights, bin_counts, )  # 画出直方图

    # 得到频数最大的bin，得到最大bin前后范一个的bin,将此三个直方图内的高程继续做一次直方图，范围更小
    max_counts = np.max(counts)
    max_index = np.where(counts == max_counts)[0][0] + 1
    min_range = max_index - 2
    max_range = max_index + 2
    if min_range < 0:
        min_range = 0
    if max_range > (len(bin_wideth) - 1):
        max_range = len(bin_wideth) - 1
    max_height = np.max(bin_wideth[min_range:max_range])
    min_height = np.min(bin_wideth[min_range:max_range])

    df03_xs = np.where((df03_heights <= max_height) & (df03_heights >= min_height), df03_xs, np.nan)
    df03_heights = np.where((df03_heights <= max_height) & (df03_heights >= min_height), df03_heights, np.nan)

    df03_xs = df03_xs[~np.isnan(df03_xs)]
    df03_heights = df03_heights[~np.isnan(df03_heights)]

    """
    v_min, v_max = binary_image_stretching(df03_heights, 1)

    df03_xs = np.where((df03_heights <= v_max) & (df03_heights >= v_min), df03_xs, np.nan)
    df03_heights = np.where((df03_heights <= v_max) & (df03_heights >= v_min), df03_heights, np.nan)

    df03_xs = df03_xs[~np.isnan(df03_xs)]
    df03_heights = df03_heights[~np.isnan(df03_heights)]
    """

    plt.close()

    return df03_xs, df03_heights


def rectangular_density_threshold_method(df03_xs, df03_heights, length=35, width=0.4, threshold=10):
    """
    根据水上光子的特性，使用长方形密度阈值法，计算以一个点为中心的长方形内的光子数量作为密度，适用阈值筛选信号光子
    :param df03_xs: 光子的沿轨距离
    :param df03_heights:  光子的高程
    :param Length: 长方形的长
    :param width:  长方形的宽
    :param threshold:  阈值
    :return: 返回高程
    """
    print("rectangular_density_threshold_method")
    density_list = []  # 每个光子的密度列表
    coordinate = np.stack([df03_xs, df03_heights], axis=1)  # 组合为坐标形式
    for i in tqdm(range(len(coordinate))):  # 逐个计算光子
        density = 0
        for j in range(len(coordinate)):
            if abs(coordinate[i][0] - coordinate[j][0]) <= length and abs(
                    coordinate[i][1] - coordinate[j][1]) < width:  # 筛选在长方体内光子
                density += 1  # 计算在长方形内的光子数量
        density_list.append(density)

    density_list = np.asarray(density_list)  # 列表转numpy数组
    df03_xs = np.asarray(df03_xs)
    df03_heights = np.asarray(df03_heights)
    df03_xs = np.where(density_list >= threshold, df03_xs, np.nan)  # 筛选密度大于threhold的信号光子
    df03_heights = np.where(density_list >= threshold, df03_heights, np.nan)
    df03_xs = df03_xs[~np.isnan(df03_xs)]
    df03_heights = df03_heights[~np.isnan(df03_heights)]

    return df03_xs, df03_heights


def ATL13_elevation(ICESat_path, NDWI_path):
    """
    使用ATL13提取水上光子
    :param ICESat_path: ICESat-2的ATL13的文件路径
    :param NDWI_path: NDWI水体的文件路径
    :return:
    """
    GP_information, pg_information = get_tif_information(NDWI_path)
    lat_lon_info = [[33.1, 111], [32.5, 111.8]]  # [[29.75, 115.7], [28.67, 116.75]]  # 湖泊范围的纬经度信息，左上角坐标，右下角坐标
    beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']

    NDWI = export_band(NDWI_path, False)  # 读取水体数据

    row_col = []
    for lonlat in lat_lon_info:
        x, y, _ = lontat_to_xy(lonlat[1], lonlat[0], pg_information[0], pg_information[1])
        row, col = xy_to_rowcol(x, y, GP_information[0])
        row_col.append([row, col])
    shape_x_y = NDWI.shape
    f = h5py.File(ICESat_path, 'r')

    row_col_h = []  # 存储行列和高程
    df03_height = []  # 存储高程

    for ii in beams:

        print(ii)
        df03 = rd.getATL13(f, ii)
        df03 = df03[df03.lats < lat_lon_info[0][0]]  # 得到坐标范围内的光子
        df03 = df03[lat_lon_info[1][0] < df03.lats]
        df03 = df03[df03.lons < lat_lon_info[1][1]]
        df03 = df03[lat_lon_info[0][1] < df03.lons]
        df03lat = df03['lats'].values
        df03lon = df03['lons'].values
        df03fullheight = df03['heights'].values

        if len(df03fullheight) >= 50:
            df03_heights = []  # 分别存储每个波束的光子的高程
            df03_lats = []
            df03_lons = []
            print("筛选水体光子", len(df03fullheight))
            # 筛选 水上光子
            for i, (lon, lan, h) in enumerate(
                    zip(df03lon, df03lat, df03fullheight,
                        )):  # enumerate用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                x, y, _ = lontat_to_xy(lon, lan, pg_information[0], pg_information[1])  # 经纬度坐标转为地理坐标
                row, col = xy_to_rowcol(x, y, GP_information[0])  # 地理坐标转为图上坐标
                if int(row - row_col[0][0] - 1) < shape_x_y[0] and int(col - row_col[0][1] - 1) < shape_x_y[1] and NDWI[
                    int(row - row_col[0][0] - 1), int(col - row_col[0][1] - 1)] == 0:  # 筛选在水体（二值化中图中，像素在0的部分）的光子
                    row_col_h.append(
                        [int(row - row_col[0][0] - 1), int(col - row_col[0][1] - 1), h])  # 添加在水体中光子的像素位置和高程
                    df03_lats.append(lan)
                    df03_lons.append(lon)
                    df03_heights.append(h)  # 添加高程值，每个波束的为一个列表

            # if len(df03_heights) > 50:  # 只有当水上光子超过50才能进入高程提取
            print(f" {ii} 的高程为：", np.mean(df03_heights) + 1.717 - 0.32)
            df03_height.append(np.mean(df03_heights) + 1.717 - 0.32)
    print(np.mean(df03_height))


def ATL08_elevation(ATL03_path, ATL08_path, NDWI_path, lat_lon_info):
    """
    使用ATL08计算高程
    :param ATL03_path: ATL03文件路径
    :param ATL08_path: ATL08文件路径
    :param NDWI_path: NDWI文件路径
    :return:
    """
    GP_information, pg_information = get_tif_information(NDWI_path)
    beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    date = ATL03_path.split("/")[3][6:14]
    NDWI = export_band(NDWI_path, False)  # 读取水体数据

    row_col = []
    for lonlat in lat_lon_info:
        x, y, _ = lontat_to_xy(lonlat[1], lonlat[0], pg_information[0], pg_information[1])
        row, col = xy_to_rowcol(x, y, GP_information[0])
        row_col.append([row, col])
    shape_x_y = NDWI.shape
    f_03 = h5py.File(ATL03_path, 'r')
    f_08 = h5py.File(ATL08_path, 'r')
    height = []
    for ii in beams:
        # ii = 'gt2r'
        print(ii)
        try:
            df03 = rd.getATL03(f_03, ii)
        except Exception as e:
            print(e)
            print("数据量太大")
            continue
        df03 = df03[df03.lats < lat_lon_info[0][0]]  # 得到坐标范围内的光子
        df03 = df03[lat_lon_info[1][0] < df03.lats]
        df03 = df03[df03.lons < lat_lon_info[1][1]]
        df03 = df03[lat_lon_info[0][1] < df03.lons]
        df03_fullx = df03['x'].values
        df03_segmentid = df03['segment_id'].values
        df03_uni_segmentid = np.unique(df03_segmentid)  # 得到数据段唯一的值
        if len(df03_uni_segmentid) <= 3:  # 若经过湖面的光子段数小于3，则不进行计算
            continue
        df03_uni_segmentid = np.delete(df03_uni_segmentid, 0)  # 删除第一个值
        df03_uni_segmentid = np.delete(df03_uni_segmentid, -1)  # 删除最后一个值

        land_height = np.asarray([])
        land_lat = np.asarray([])
        land_lon = np.asarray([])
        dist_ph_along = np.asarray([])  #沿轨距离

        df08 = rd.getATL08(f_08, ii)
        for segmentid in df03_uni_segmentid:
            df08_date = df08[df08.ph_segment_id == segmentid]
            if len(df08_date) != 0:
                df08_date = df08_date[df08.pc_flag == 1]
                if len(df08_date) == 0:
                    continue
            else:
                continue
            df03_date = df03[df03.segment_id == segmentid]
            if len(df03_date) == 0:
                continue
            index = df08_date['pc_indx'].values
            if np.max(index) + 1 > len(df03_date):
                print("发生错误！")
                print("df08段内光子数为：", np.max(index) + 1)
                print("df03段内光子数为：", len(df03_date))
                print("段的编号为：", segmentid)
                continue
            index += np.min(df03_date.index)
            try:
                df03_date = df03_date.loc[index]
            except Exception as e:
                print(f"发生错误: {e}",f"段的编号为：{segmentid}")

            land_height = np.append(land_height, df03_date['heights'].values)
            land_lat = np.append(land_lat, df03_date['lats'].values)
            land_lon = np.append(land_lon, df03_date['lons'].values)
            dist_ph_along = np.append(dist_ph_along, df03_date['x'].values)

        water_heights = []  # 分别存储每个波束的光子的高程

        print("筛选水体光子", len(df03_fullx))
        # 筛选 水上光子
        for i, (lon, lan, h) in enumerate(
                zip(land_lon, land_lat, land_height
                    )):  # enumerate用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            x, y, _ = lontat_to_xy(lon, lan, pg_information[0], pg_information[1])  # 经纬度坐标转为地理坐标
            row, col = xy_to_rowcol(x, y, GP_information[0])  # 地理坐标转为图上坐标
            if int(row - row_col[0][0] - 1) < shape_x_y[0] and int(col - row_col[0][1] - 1) < shape_x_y[1] and NDWI[
                int(row - row_col[0][0] - 1), int(col - row_col[0][1] - 1)] == 0:  # 筛选在水体（二值化中图中，像素在0的部分）的光子

                water_heights.append(h)  # 添加高程值，每个波束的为一个列表
        if len(water_heights) > 50:  # 只有当水上光子超过50才能进入去噪和高程提取
            water_heights = np.asarray(water_heights)  # 转为numpy数组

            height.append(np.mean(water_heights + 1.717 - 0.32))  #坐标改正
            print(f"{ii}的高程为：", np.mean(water_heights))

        del df08
        del df03
    print(f"{date}的最终高程为：{np.mean(height)}")

lake_file = "富水水库"  # 水库名
ATL03_paths = f"F:/吕鑫/湖泊原始数据/斧头湖/ATL03"  # 必须只有ICESat-2数据
ATL08_paths = f"F:/吕鑫/湖泊原始数据/斧头湖/ATL08"  # 必须只有ICESat-2数据
NDWI_paths = f"F:/吕鑫/湖泊原始数据/斧头湖/processed_NDWI"  # 必须只有水体的tiff数据
lat_lon_info = [[30.1451, 114.1493], [29.9614, 114.3182]]
ATL03_files = os.listdir(ATL03_paths)
ATL08_files = os.listdir(ATL08_paths)
NDWI_files = os.listdir(NDWI_paths)
ATL03_ATL08_bool = False  # 用来判断ATL03与ATL08是否有对应的数据
for ATL03_file in ATL03_files:
    ATL03_date = ATL03_file.split("_")[1]
    ATL03_datetime = datetime.date(int(ATL03_date[0:4]), int(ATL03_date[4:6]), int(ATL03_date[6:8]))
    for ATL08_file in ATL08_files:
        ATL08_date = ATL08_file.split("_")[1]
        ATL08_datetime = datetime.date(int(ATL08_date[0:4]), int(ATL08_date[4:6]), int(ATL08_date[6:8]))
        if ATL08_datetime == ATL03_datetime:
            min_date = 1000
            for NDWI_file in NDWI_files:
                NDWI_date = NDWI_file.split("_")[0]
                NDWI_datetime = datetime.date(int(NDWI_date[0:4]), int(NDWI_date[4:6]), int(NDWI_date[6:8]))
                date = abs((ATL03_datetime - NDWI_datetime).days)
                if date < min_date:  # 筛选Landsat水体离光子数据最近的水体
                    min_date = date
                    min_NDWI_date = NDWI_date[0:8]
                    suitable_NDWI = NDWI_file
            if min_date < 30:  # 若最近日期在一个月之外，则放弃
                print(min_date, suitable_NDWI, ATL03_datetime, ATL08_datetime)

                # 开始计算水体高程
                ATL03_path = ATL03_paths + "/" + ATL03_file
                ATL08_path = ATL08_paths + "/" + ATL08_file
                NDWI_path = NDWI_paths + "/" + suitable_NDWI
                ATL08_elevation(ATL03_path, ATL08_path, NDWI_path, lat_lon_info)


"""
ATL03_path = "./2021/2021_ICESat-2/ATL03/ATL03_20210108170032_02401002_006_01.h5"
ATL08_path = "./2021/2021_ICESat-2/ATL08/ATL08_20210108170032_02401002_006_01.h5"
NDWI_path = "./2021/2021_image_processing/20210108_NDWI.tif"
GP_information, pg_information = get_tif_information(NDWI_path)
lat_lon_info = [[33.1, 111], [32.5, 111.8]]  # [[29.75, 115.7], [28.67, 116.75]]  # 湖泊范围的纬经度信息，左上角坐标，右下角坐标
beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
date = ATL03_path.split("/")[3][6:14]
NDWI = export_band(NDWI_path, False)  # 读取水体数据
plt.figure(0)
plt.imshow(NDWI, cmap='gray')

row_col = []
for lonlat in lat_lon_info:
    x, y, _ = lontat_to_xy(lonlat[1], lonlat[0], pg_information[0], pg_information[1])
    row, col = xy_to_rowcol(x, y, GP_information[0])
    row_col.append([row, col])
shape_x_y = NDWI.shape
f_03 = h5py.File(ATL03_path, 'r')
f_08 = h5py.File(ATL08_path, 'r')
height = []
n = 1
#for ii in beams:
ii = 'gt2r'
print(ii)
df03 = rd.getATL03(f_03, ii)
df03 = df03[df03.lats < lat_lon_info[0][0]]  # 得到坐标范围内的光子
df03 = df03[lat_lon_info[1][0] < df03.lats]
df03 = df03[df03.lons < lat_lon_info[1][1]]
df03 = df03[lat_lon_info[0][1] < df03.lons]
df03lat = df03['lats'].values
df03lon = df03['lons'].values
df03fullheight = df03['heights'].values + 1.717 - 0.32
df03_fullx = df03['x'].values
df03_segmentid = df03['segment_id'].values
df03_uni_segmentid = np.unique(df03_segmentid)  # 得到数据段唯一的值
df03_uni_segmentid = np.delete(df03_uni_segmentid, 0)  # 删除第一个值
df03_uni_segmentid = np.delete(df03_uni_segmentid, -1)  # 删除最后一个值

land_x = np.asarray([])
land_height = np.asarray([])
land_lat = np.asarray([])
land_lon = np.asarray([])

df08 = rd.getATL08(f_08, ii)
for segmentid in df03_uni_segmentid:
    df08_date = df08[df08.ph_segment_id == segmentid]
    if len(df08_date) != 0:
        df08_date = df08_date[df08.pc_flag == 1]
        if len(df08_date) == 0:
            continue
    else:
        continue
    df03_date = df03[df03.segment_id == segmentid]
    if len(df03_date) == 0:
        continue
    index = df08_date['pc_indx'].values
    if np.max(index) + 1 > len(df03_date):
        print("发生错误！")
        print("df08段内光子数为：", np.max(index) + 1)
        print("df03段内光子数为：", len(df03_date))
        print("段的编号为：", segmentid)
        continue
    index += np.min(df03_date.index)
    df03_date = df03_date.loc[index]
    land_x = np.append(land_x, df03_date['x'].values)
    land_height = np.append(land_height, df03_date['heights'].values)
    land_lat = np.append(land_lat, df03_date['lats'].values)
    land_lon = np.append(land_lon, df03_date['lons'].values)
n += 1
plt.figure(n)
plt.scatter(df03_fullx, df03fullheight, s=1)
plt.scatter(land_x, land_height, s=1, c="red")
io.savemat('full_photon.mat', {'x': df03_fullx, 'y': df03fullheight})
io.savemat('land_photon.mat', {'x': land_x, 'y': land_height})

water_heights = []  # 分别存储每个波束的光子的高程
water_xs = []  # 分别存储每个波束的光子的沿轨距离
water_lats = []
water_lons = []

print("筛选水体光子", len(df03_fullx))
# 筛选 水上光子
for i, (lon, lan, h, xs) in enumerate(
        zip(land_lon, land_lat, land_height,
            land_x)):  # enumerate用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    x, y, _ = lontat_to_xy(lon, lan, pg_information[0], pg_information[1])  # 经纬度坐标转为地理坐标
    row, col = xy_to_rowcol(x, y, GP_information[0])  # 地理坐标转为图上坐标
    if int(row - row_col[0][0] - 1) < shape_x_y[0] and int(col - row_col[0][1] - 1) < shape_x_y[1] and NDWI[
        int(row - row_col[0][0] - 1), int(col - row_col[0][1] - 1)] == 0:  # 筛选在水体（二值化中图中，像素在0的部分）的光子
        plt.figure(0)
        plt.scatter(int(col - row_col[0][1] - 1), int(row - row_col[0][0] - 1), c="red")
        water_lats.append(lan)
        water_lons.append(lon)
        water_heights.append(h)  # 添加高程值，每个波束的为一个列表
        water_xs.append(xs)
if len(water_xs) > 50:  # 只有当水上光子超过50才能进入去噪和高程提取
    io.savemat('water_photon_08.mat', {'x': water_xs, 'y': water_heights})
    water_heights = np.asarray(water_heights)  # 转为numpy数组
    water_xs = np.asarray(water_xs)

    n = n + 1
    plt.figure(n)
    plt.title(f"water in land {ii}")
    plt.scatter(land_x, land_height, s=1)  # 区域内的所用光子
    plt.scatter(water_xs, water_heights, s=1, c="red")
    vmin, vmax, water_heights = binary_image_stretching(water_heights, 3)
    print(vmin, vmax)
    height.append(np.mean(water_heights) + 1.717 - 0.32)
    print(f"{ii}的高程为：", np.mean(water_heights) + + 1.717 - 0.32)
print(f"{date}最终高程为：", np.mean(height))
plt.show()
"""
