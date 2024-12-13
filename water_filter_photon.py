"""
根据区域得到水体掩膜上的水上光子
"""
import os  # 文件处理库
import numpy as np  # 科学计算库 
from osgeo import gdal, osr  # 处处理栅格数据的库
import h5py
import readers as rd  # 读取光子的函数
import datetime
from tqdm import tqdm


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



def  ATL03_filter(ICESat_path, NDWI_path, lat_lon_info, date, save_path):
    """
    使用ATL03，直方图去噪得到水位光子
    :param ICESat_path: ICESat-2的ATL03的文件路径
    :param NDWI_path: NDWI水体的文件路径
    :return:
    """
    GP_information, pg_information = get_tif_information(NDWI_path)
    beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']

    NDWI = export_band(NDWI_path, False)  # 读取水体数据

    row_col = []
    for lonlat in lat_lon_info:
        x, y, _ = lontat_to_xy(lonlat[1], lonlat[0], pg_information[0], pg_information[1])
        row, col = xy_to_rowcol(x, y, GP_information[0])
        row_col.append([row, col])
    shape_x_y = NDWI.shape
    f = h5py.File(ICESat_path, 'r')

    for ii in beams:

        """用于判断文件是否存在"""
        date_folder = os.path.join(save_path, date)  # 创建新文件夹存储数据
        file_path = os.path.join(date_folder, f"data_{ii}.csv")
        if os.path.exists(file_path):
            continue
        else:
            print(f"File {file_path} does not exist.")

            print(ii)
            #try:
            df03 = rd.getATL03(f, ii)
            #except Exception as e:
                #print(e)
                #continue
            df03 = df03[df03.lats < lat_lon_info[0][0]]  # 得到坐标范围内的光子
            df03 = df03[lat_lon_info[1][0] < df03.lats]
            df03 = df03[df03.lons < lat_lon_info[1][1]]
            df03 = df03[lat_lon_info[0][1] < df03.lons]

            if len(df03) >= 50:
                min_x = np.min(df03['x'])
                df03['x'] = df03['x'] - min_x

                mask = np.full(len(df03), False, dtype=bool)
                for idx, (index, row) in tqdm(enumerate(df03.iterrows()), total=len(df03), desc="Processing", position=0, leave= True):
                    lon = row['lons']
                    lat = row['lats']

                    x, y, _ = lontat_to_xy(lon, lat, pg_information[0], pg_information[1])  # 经纬度坐标转为地理坐标
                    row, col = xy_to_rowcol(x, y, GP_information[0])  # 地理坐标转为图上坐标
                    if int(row - row_col[0][0] - 1) < shape_x_y[0] and int(col - row_col[0][1] - 1) < shape_x_y[1] and NDWI[
                        int(row - row_col[0][0] - 1), int(col - row_col[0][1] - 1)] == 0:  # 筛选在水体（二值化中图中，像素在0的部分）的光子
                        mask[idx] = True

                df03 = df03[mask]

                date_folder = os.path.join(save_path, date)  # 创建新文件夹存储数据
                if not os.path.exists(date_folder):
                    os.makedirs(date_folder)
                else:
                    print("1")

                file_path = os.path.join(date_folder, f"data_{ii}.csv")
                df03.to_csv(file_path, index=False)
                print("\033[91m没有的已存入\033[0m")  # 显示为红色更醒目
                del df03



ICESats = f"G:/吕鑫/纳木错/ICESat-2数据过大"   # 必须只有ICESat-2数据
NDWIWater = f"D:/研究生/学校/大论文/纳木错流域湖泊数据/未命名2/未命名2湖泊/图像后处理/挑选好的影像"  # 必须只有水体的tiff数据
save_path = f"D:/研究生/学校/大论文/纳木错流域湖泊数据/未命名2/未命名2水位/水上光子"
lat_lon_info = [[31.844024, 90.316887], [31.706556, 90.490265]]
ICESat_files = os.listdir(ICESats)
NDWI_files = os.listdir(NDWIWater)
for ICESat_file in ICESat_files:
    ICESat_date = ICESat_file.split("_")[1]
    ICESat_datetime = datetime.date(int(ICESat_date[0:4]), int(ICESat_date[4:6]), int(ICESat_date[6:8] ))
    min_date = 1000
    for NDWI_file in NDWI_files:
        NDWI_date = NDWI_file.split("_")[0]
        NDWI_datetime = datetime.date(int(NDWI_date[0:4]), int(NDWI_date[4:6]), int(NDWI_date[6:8]))
        date = abs((ICESat_datetime - NDWI_datetime).days)
        if date < min_date:  # 筛选Landsat水体离光子数据最近的水体
            min_date = date
            min_NDWI_date = NDWI_date[0:8]
            suitable_NDWI = NDWI_file
    date = ICESat_file[6:10] + "." + ICESat_file[10:12] + "." + ICESat_file[12:14]
    print(ICESat_file, suitable_NDWI, min_date, ICESat_file[6:10] + "." + ICESat_file[10:12] + "." + ICESat_file[12:14])

    # 开始提取水上光子
    ICESat_path = ICESats + "/" + ICESat_file
    NDWI_path = NDWIWater + "/" + suitable_NDWI
    ATL03_filter(ICESat_path, NDWI_path, lat_lon_info, date, save_path)

