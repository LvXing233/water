import numpy as np
import netCDF4 as nc
from osgeo import gdal, osr, ogr
import os
import glob
import datetime
import math

"""将所有的月度蒸发量tiff文件加起来合并为年度的"""
def combine_monthly_to_annual(Input_folder, Output_folder, year):
    # 获取所有月度tiff文件路径，假设文件名包含月份信息，如 etp_201801.tif, etp_201802.tif, etc.
    monthly_tiffs = sorted(glob.glob(os.path.join(Input_folder, f"{year}*.tif")))

    if len(monthly_tiffs) != 12:
        print(f"Warning: Expected 12 months of data, but found {len(monthly_tiffs)} files.")

    # 读取第一个tiff文件，获取影像大小和投影信息
    first_tif = gdal.Open(monthly_tiffs[0])
    N_Lat = first_tif.RasterYSize
    N_Lon = first_tif.RasterXSize

    # 获取地理信息
    geotransform = first_tif.GetGeoTransform()
    projection = first_tif.GetProjection()

    # 初始化一个空的数组来存储年度总和
    annual_sum = np.zeros((N_Lat, N_Lon), dtype=np.float32)

    # 遍历每个月的tiff文件并累加值
    for tif_file in monthly_tiffs:
        # 读取每个月的tiff文件
        tif = gdal.Open(tif_file)
        band = tif.GetRasterBand(1)
        data = band.ReadAsArray()  # 读取影像数据

        # 处理缺失值（如果有的话，假设-32768是无效值，调整为NaN）
        data[data == -32768] = np.nan

        # 将当前月数据累加到年度总和
        annual_sum += data

    # 创建一个输出年度总和的tiff文件
    driver = gdal.GetDriverByName('GTiff')
    output_file = os.path.join(Output_folder, f"annual_etp_{year}.tif")
    out_tif = driver.Create(output_file, N_Lon, N_Lat, 1, gdal.GDT_Float32)

    # 设置地理信息
    out_tif.SetGeoTransform(geotransform)
    out_tif.SetProjection(projection)

    # 写入年度总和数据
    out_tif.GetRasterBand(1).WriteArray(annual_sum)
    out_tif.FlushCache()  # 保存文件

    print(f"年度总和tiff文件创建成功: {output_file}")


def main():
    Input_folder = r'G:/MT/tmp_2022'  # 月度tiff文件的输入文件夹
    Output_folder = r'G:/MT/tmp_2022'  # 输出年度tiff文件的文件夹

    # 设置年份
    year = 2022  # 你要合并的年份

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(Output_folder):
        os.makedirs(Output_folder)

    # 合并每月数据为年度数据
    combine_monthly_to_annual(Input_folder, Output_folder, year)


if __name__ == '__main__':
    main()
