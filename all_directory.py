# 水上光子去噪示意
import copy
from tqdm import tqdm
import numpy as np  # 科学计算库
import pandas as pd
import matplotlib.pyplot as plt  # 绘图库
from sklearn.neighbors import KDTree
import os
import hdbscan  # HDBSCAN官方算法

# 分段去噪
"""
计算一个文件夹里的所有水上光子的csv文件进行去噪
"""


def read_csv(csv_path):
    """
    读取csv文件
    @param csv_path: csv的文件
    """
    data = pd.read_csv(csv_path)

    return data


def plot_photon_1(data, num, s=2):
    """
    画出水上光子图
    @param data: pandas的表
    @param num: 画在第几幅画上
    """
    plt.figure(num)
    plt.scatter(data['x'], data['heights'], s=s, label="Removed photons")


def plot_photon_2(x, height, num, s=2, col=None):
    """
    画出水上光子图
    @param data: pandas的表
    @param num: 画在第几幅画上
    """
    plt.figure(num)
    if col:
        plt.scatter(x, height, s=s, label="Effective signal photons", color=col)
    else:
        plt.scatter(x, height, s=s, label="Effective signal photons")


def plot_histogram(data, num):
    """
    科学方法画出高程直方图
    @param data: pandas的表
    @param num: 画在第几幅画上
    """
    plt.figure(num)
    bin_counts = int(np.sqrt(len(data)))
    counts, bin_wideth = np.histogram(data['height'], bin_counts)
    plt.hist(data['heights'], bin_counts, color="blue")

    return counts, bin_wideth


def detect_outliers_iqr(data, threshold=2):
    """
    使用箱线图检测异常值
    :param data:numpy数组
    :param threshold:IQR的倍数
    :return:返回最大值的阈值
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)
    return upper_bound

def filter_hright(all_height):
    """
    用于高程异常值去除，减少工作量
    :param all_height: 高程值
    :return:
    """
    #去除nan值
    all_height = all_height[~np.isnan(all_height)]
    if all_height.size == 0:  # 如果全是空值，则返回空数组
        return np.array([])
    # 计算中位数和四分位数
    median_height = np.median(all_height)
    Q1 = np.percentile(all_height, 25)
    Q3 = np.percentile(all_height, 75)
    IQR = Q3 - Q1

    # 定义异常值的范围
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 过滤掉异常值
    filtered_heights = all_height[(all_height >= lower_bound) & (all_height <= upper_bound)]

    return filtered_heights

def calculate_height(df03):
    """
    计算水位高程
    :param df03: df03为表头为x,heights的pandas数据格式
    :return:返回这个轨道的信号光子
    """
    min_x = df03['x'].min()
    max_x = df03['x'].max()

    all_height = np.array([])

    segment_x = min_x + 100
    segment_data = df03[df03.x > min_x]
    segment_data = segment_data[segment_data.x < segment_x]

    heights = segment_data['heights'].values  # numpy数组
    median_height = np.median(heights)

    all_height = np.append(all_height, median_height)

    while segment_x < max_x:  # 分段对光子进行计算
        min_x = segment_x
        segment_x = min_x + 100
        segment_data = df03[df03.x > min_x ]
        segment_data = segment_data[segment_data.x < segment_x]

        heights = segment_data['heights'].values  # numpy数组
        median_height = np.median(heights)

        all_height = np.append(all_height, median_height)

    all_height = filter_hright(all_height)

    if all_height.size == 0:  # 如果为空数组，说明没有值
        return None

    height = np.mean(all_height)

    return height


def chance(all_tree, leaf1=None, leaf2=None, num=None):
    """
    用来修改聚类树的最后的值，为类添加标志
    @param all_tree: 整体聚类树
    @param leaf1: 簇的左第一个起始边
    @param leaf2: 簇的右最后起始边
    @param num: 聚类簇的标识，-1为噪声
    @return:
    """
    if leaf2 == None:
        index = all_tree.index(leaf1)
        leaf1.append(num)
        all_tree[index] = leaf1
    else:
        index1 = all_tree.index(leaf1)
        index2 = all_tree.index(leaf2)
        for leaf in all_tree[index2:index1 + 1]:
            leaf.append(num)


def calculate_Divisive(tree, all_tree=None):
    """
    计算分裂值
    :param tree:
    :return:
    """
    """
       计算分裂值
       :param tree:
       :return:
       """
    divisive_value = []
    if all_tree == None:
        all_tree = tree  # 使用all_tree作为副本，能够更改tree里的值，以后的值只能修改all_tree，再改tree

    for leaf in tree:
        divisive_value.append(leaf[2])  # 统计叶子长度

    std_deviation = np.std(divisive_value)  # 计算标准差

    divisive_value = np.array(divisive_value)
    threhold = detect_outliers_iqr(divisive_value)

    for leaf in tree:
        if leaf[2] > threhold:  # 先简单以距离对数据进行去噪
            chance(all_tree, leaf, num=-1)

    for leaf in tree:
        if leaf[2] > 3 * std_deviation and leaf[3] == 1:  # 先简单以距离对数据进行去噪
            # 左节点
            index1 = tree.index(leaf[4])
            index2 = tree.index(leaf[5])
            left_leaf = tree[index2:index1 + 1]  # 左簇
            left_number = len(left_leaf) + 1  # 粗略计算节点数量【包括了分裂节点】

            # 右节点
            index3 = tree.index(leaf[6])
            index4 = tree.index(leaf[7])
            right_leaf = tree[index4:index3 + 1]  # 右簇
            right_number = len(right_leaf) + 1  # 粗略计算节点数量【包括了分裂节点】

            if left_number == 1 or right_number == 1:
                continue

            if right_number <= 4:
                leaf1 = tree[index3]
                leaf2 = tree[index4]
                chance(all_tree, leaf1, leaf2, -1)

            if left_number <= 4:
                leaf1 = tree[index1]
                leaf2 = tree[index2]
                chance(all_tree, leaf1, leaf2, -1)
    return


def condensed(tree, size, higher_level_cluster_stability=0, num=0, cluster_leaf1=None, cluster_leaf2=None,
              all_tree=None):
    """
    @param tree: 最小生成树
    @param tree: 最小簇的大小
    @param higher_level_cluster_stability: 上一层的聚类簇的稳定性，没有就默认为0
    @param num: 属于哪一类的标识
    @param cluster_leaf1: 一个节点下类的起始边的索引
    @param cluster_leaf2: 一个节点下类的终止边的索引
    @return:
    """
    """对聚类数进行剪枝，去除散点和类簇小于5的节点"""
    if all_tree == None:
        all_tree = tree  # 使用all_tree作为副本，能够更改tree里的值，以后的值只能修改all_tree，再改tree
    for leaf in tree:
        if leaf[3] == 0:  # 初始筛选掉最远的散点
            chance(all_tree, leaf, num=-1)
        elif leaf[3] == 1:  # 分裂为两个簇
            break


def tree_edge_statistics(X, edges):
    """
    用于统计边的长，横轴距离和纵轴距离，并返回加权后的边
    :param X: 数据点
    :param edges:最小生成树的边
    :return:返回加权后的边
    """
    sta_edges = copy.deepcopy(edges)  # 不修改原来的值

    # 对每个边添加横轴和纵轴距离
    x_distance = []
    y_distance = []
    for i in range(len(sta_edges)):
        edge = sta_edges[i]
        point1 = X[edge[0]]
        point2 = X[edge[1]]
        x_dis = abs(point1[0] - point2[0])
        y_dis = abs(point1[1] - point2[1])
        x_distance.append(x_dis)
        y_distance.append(y_dis)
        edge.append(x_dis)
        edge.append(y_dis)
        sta_edges[i] = edge


def Change_tree_sides(X, edges):
    """
    将聚类树的边换为垂直边，但是之前形成边的依据是欧氏距离
    :param X: 数据点
    :param edges:欧氏距离得到的边
    :return:返回转为垂直距离的边的边
    """
    sta_edges = copy.deepcopy(edges)  # 不修改原来的值
    for i in range(len(sta_edges)):
        edge = sta_edges[i]
        point1 = X[edge[0]]
        point2 = X[edge[1]]
        y_dis = abs(point1[1] - point2[1])
        edge[2] = y_dis  # 这样也能将列表的值进行修改

    return sta_edges


def Calculate_UWG(X, num=1):
    """
    生成无向图，各个点的距离，默认为HDBSCAN定义的 互可达距离
    :param X:numpy二维数组，两列N行，每一行为坐标
    :return:返回无向图的距离矩阵
    """
    if num == 1:
        # 计算无向加权图，互可达距离矩阵，默认欧氏距离作为核心距离
        tree = KDTree(X)  # 使g    用KDtree进行检索
        dist, ind = tree.query(X, k=5)  # 检索邻近五个点的距离
        core_distance = np.max(dist, axis=1)  # 得到最远的五个点的距离作为核心距离

        UWG = np.full((X.shape[0], X.shape[0]), np.inf)  # 各个点的互可达距离，用于生成最小生成树
        # 计算无向加权图，互可达距离矩阵
        for i in range(X.shape[0]):
            a_coredis = core_distance[i]
            for j in range(X.shape[0]):
                b_coredis = core_distance[j]
                distance = np.sqrt(np.sum(np.square(X[i] - X[j])))
                if i != j:
                    reachability_distance = np.max(
                        [a_coredis, b_coredis, distance])  # 得到互可达距离，d(a,b)=max(core(a),core(b),dist(a,b))
                    UWG[i, j] = reachability_distance

    elif num == 2:
        # 计算无向加权图，互可达距离矩阵，以垂直距离作为核心距离
        tree = KDTree(X)  # 使g    用KDtree进行检索
        dist, ind = tree.query(X, k=5)  # 检索邻近五个点的距离，索引
        core_distance = np.array([])
        for i in range(ind.shape[0]):  # 顺序得到数组里
            points = X[ind[i]]  # 得到最近五个点
            max_Vertical_Distance = 0
            for j in range(points.shape[0]):
                Vertical_Distance = abs(X[i][1] - points[j][1])  # 计算垂直距离
                if max_Vertical_Distance < Vertical_Distance:
                    max_Vertical_Distance = Vertical_Distance
            core_distance = np.append(core_distance, max_Vertical_Distance)  # 将临近五个点的距离核心点的最大垂直距离作为核心距离

        UWG = np.full((X.shape[0], X.shape[0]), np.inf)  # 各个点的互可达距离，用于生成最小生成树
        # 计算每个点之间相互的互可达距离（PS:垂直距离为基准）
        for i in range(X.shape[0]):
            a_coredis = core_distance[i]  # 提取第一个点的核心距离
            for j in range(X.shape[0]):
                b_coredis = core_distance[j]  # 提取第二个点的核心距离
                Vertical_Distance = abs(X[i][1] - X[j][1])  # 计算垂直距离
                if i != j:
                    reachability_distance = np.max(
                        [a_coredis, b_coredis, Vertical_Distance])  # 得到互可达距离，d(a,b)=max(core(a),core(b),dist(a,b))
                    UWG[i, j] = reachability_distance
    elif num == 3:
        # 计算无向加权图，互可达距离矩阵，减少横轴上的距离差异，以增大离散点的欧氏距离与横轴上离散距离的差异
        tree = KDTree(X)  # 用KDtree进行检索
        dist, ind = tree.query(X, k=5)  # 检索邻近五个点的距离，索引
        core_distance = np.array([])
        for i in range(ind.shape[0]):  # 顺序得到数组里
            point = X[ind[i][4]]  # 得到最近的第五个点
            x = X[i][0] - point[0]
            y = X[i][1] - point[1]
            x = x / 4  # 减少横轴的距离
            distance = np.sqrt(x ** 2 + y ** 2)
            core_distance = np.append(core_distance, distance)

        UWG = np.full((X.shape[0], X.shape[0]), np.inf)  # 各个点的互可达距离，用于生成最小生成树
        # 计算无向加权图，互可达距离矩阵
        for i in range(X.shape[0]):
            a_coredis = core_distance[i]
            for j in range(X.shape[0]):
                b_coredis = core_distance[j]
                x = X[i][0] - X[j][0]
                y = X[i][1] - X[j][1]
                x = x / 4
                distance = np.sqrt(x ** 2 + y ** 2)
                if i != j:
                    reachability_distance = np.max(
                        [a_coredis, b_coredis, distance])  # 得到互可达距离，d(a,b)=max(core(a),core(b),dist(a,b))
                    UWG[i, j] = reachability_distance

    elif num == 4:
        # 计算无向加权图，互可达距离矩阵，选取两个点作，即最近的一个点作为核心距离
        tree = KDTree(X)  # 用KDtree进行检索
        dist, ind = tree.query(X, k=2)  # 检索邻近五个点的距离，索引
        core_distance = np.max(dist, axis=1)  # 得到最远的五个点的距离作为核心距离
        UWG = np.full((X.shape[0], X.shape[0]), np.inf)  # 各个点的互可达距离，用于生成最小生成树
        # 计算无向加权图，互可达距离矩阵
        for i in range(X.shape[0]):
            a_coredis = core_distance[i]
            for j in range(X.shape[0]):
                b_coredis = core_distance[j]
                distance = np.sqrt(np.sum(np.square(X[i] - X[j])))
                if i != j:
                    reachability_distance = np.max(
                        [a_coredis, b_coredis, distance])  # 得到互可达距离，d(a,b)=max(core(a),core(b),dist(a,b))
                    UWG[i, j] = reachability_distance

    return UWG


def denoising_preprocessing(X, width=None):
    """
    用于对光子进行预处理，对其进行粗去噪
    :param X:包含横纵坐标的光子信息
    :param width:如果有就确定直方图的宽度，如果没有则使用公式得到直方图箱子的数量
    :return:返回预处理去噪后的光子
    """
    height_values = X[:, -1]  # 得到heights的值
    if width:
        min_val = np.min(height_values)
        max_val = np.max(height_values)

        # 计算直方图的边界
        bin_edges = np.arange(min_val, max_val + width, width)
        hist, _ = np.histogram(height_values, bins=bin_edges)  # 得到直方图的频数和边界
    else:
        bin_counts = int(np.sqrt(len(height_values)))  # 根据数据范围得到直方图的箱子数量
        hist, bin_edges = np.histogram(height_values, bin_counts)  # 得到直方图的频数和边界
        # 得到最大频数范围的前后一个频数的范围
    max_index = np.argmax(hist)  # 最大值的索引
    max_height_index = max_index + 1
    min_height_index = max_index
    max_height = bin_edges[max_height_index]
    min_height = bin_edges[min_height_index]  # 得到去噪后的范围
    condition = (height_values > min_height) & (height_values < max_height)
    denosing_X = X[condition]

    return denosing_X


def Data_Quality(height_values, width = None):
    """
    用韵判断数据里的信号光子质量
    :param X:高程数据
    :return:返回质量分数
    """
    if width:
        min_val = np.min(height_values)
        max_val = np.max(height_values)
        # 计算直方图的边界
        bin_edges = np.arange(min_val, max_val + width, width)
        counts, _ = np.histogram(height_values, bins=bin_edges)
    else:
        bin_counts = int(np.sqrt(len(height_values)))
        counts, _ = np.histogram(height_values, bin_counts)
    counts = counts[~np.isnan(counts) & (counts != 0)]
    if len(counts) >= 3:
        max_counts = np.max(counts)
        median_counts = np.median(counts)
    elif len(counts) == 2:
        counts = np.sort(counts)
        max_counts = counts[0]
        median_counts = counts[1]
    elif len(counts) == 1:
        return 5
    else:
        return 0

    core = max_counts / median_counts

    return core


def HDBSCAN_my(x, height, size, num=False):
    """
    python语言写的普通版本HDBSCAN
    :param x: 光子沿轨距离
    :param height: 光子高程
    :param size: 选择生成无向图的方式
    :param num: 是否画图的选项,False不画, num为数字就是过程画图
    :return: 返回信号光子
    """

    # 对数据进行拼接，处理
    X = np.column_stack((x, height))



    if len(X) <= 2:
        return np.array([])  # 小于五十个光子则认为数据无效

    # 质量分数判断
    core = Data_Quality(X[:, -1], width=3)
    if core < 8:
        return np.array([])  # 质量分数少于4的作为噪声光子

    # 对数据进行粗去噪
    X = denoising_preprocessing(X, 3)

    if len(X) <= 1:
        return np.array([])
    UWG = Calculate_UWG(X, size)  # 计算无向加权图

    # 生成prime最小生成树
    tree_nodes = [0]  # 树节点，先随机第一个
    tree_edges = []  # 树的边
    points = list(range(X.shape[0]))  # 未进入节点的点
    tree_UWG = UWG

    while len(points) != 1:
        min_edge = np.inf
        min_position = np.inf
        for tree_node in tree_nodes:
            # 得到一个点到另一个点的边的最小值和点的位置
            edge = np.min(tree_UWG[tree_node, :])
            position = np.argmin(tree_UWG[tree_node, :])
            if min_edge >= edge and edge != np.inf:
                min_edge = edge  # 最小边的权值
                min_position = position  # 最小边另一个点的位置
                previous_node = tree_node  # 最小边

        points.remove(min_position)
        for tree_node in tree_nodes:
            tree_UWG[tree_node, min_position] = np.inf
            tree_UWG[min_position, tree_node] = np.inf
        tree_nodes.append(min_position)
        tree_edges.append([previous_node, min_position, min_edge])

    # 画最小生成树图
    if num:
        num += 1
        plt.figure(num)
        for row in tree_edges:
            plt.plot(*X[row[0], :], 'ro', c="b", markersize=3)
            plt.plot(*X[row[1], :], 'ro', c="b", markersize=3)
            plt.plot([X[row[0], :][0], X[row[1], :][0]], [X[row[0], :][1], X[row[1], :][1]], 'k-', label='Connection')

    tree_edges = Change_tree_sides(X, tree_edges)
    # 进行层次聚类的步骤
    # 对树的边进行排序
    # 0代表普通叶节点，1代表连接两个类的边，2代表类的第一个节点
    sorted_tree_edges = sorted(tree_edges, key=lambda x: x[2])
    clusters = []  # 聚类结果
    while len(sorted_tree_edges) != 0:  # 循环取出所有点
        stop_outer_loop = False  # 代表for循环是否停止的bool值
        new_cluster = sorted_tree_edges[0]  # 取出一条新边
        node1 = new_cluster[0]  # 边的第一个点
        node2 = new_cluster[1]  # 边的第二个点

        if len(clusters) == 0:  # 初始化聚类
            new_cluster.append(2)  # 标记为叶节点，第一个类的第一个叶子节点
            new_cluster.append(2)  # 表示两个节点都是类中的
            clusters.append([new_cluster])  # 聚类列表为空，则加入一个新的聚类作为初始聚类【即为最小的边】
            sorted_tree_edges.pop(0)  # 将选出的边弹出
            continue
        for i in range(len(clusters)):  # 开始使用最小生成树进行聚类，取出已经分好的类
            for j in range(len(clusters[i])):  # 取出已经分好类里的边与未分类的边进行比较看是否已经有连接
                if node1 in clusters[i][j][0:2] or node2 in clusters[i][j][0:2]:  # 如果有边的点在类里
                    # 查找一个边是否是两个类的连接边
                    for k in range(len(clusters)):  # 查找是否有两个类是由边连在一起的，
                        for l in range(len(clusters[k])):
                            if i != k:  # 避免重复的类
                                if node1 in clusters[k][l][0:2] or node2 in clusters[k][l][0:2]:  # 如果有边的点在类中
                                    merge_cluster = clusters[i] + clusters[k]  # 将两个类合并
                                    new_cluster.append(1)  # 1表示是节点，表示两个类连接的边

                                    # 将两个类的起始边与结束边加入到连接两个类的边后面，用来进行节点的分裂
                                    new_cluster.append(clusters[i][0])  # 第一个类的起始边
                                    new_cluster.append(clusters[i][-1])  # 第一个类的终止边
                                    new_cluster.append(clusters[k][0])  # 第二个类的起始边
                                    new_cluster.append(clusters[k][-1])  # 第二个类的终止边

                                    merge_cluster.append(new_cluster)  # 将边加入到新的类中

                                    # 删除原来的类
                                    del clusters[i]
                                    del clusters[k - 1]
                                    clusters.insert(i, merge_cluster)  # 将新类插入

                                    sorted_tree_edges.pop(0)  # 将加入的边在结果中删除
                                    stop_outer_loop = True
                                    break  # 倒数第一个循环停止

                        if stop_outer_loop:
                            break  # 倒数第二个循环停止

                    if stop_outer_loop:
                        break  # 倒数第三个循环停止

                    new_cluster.append(0)  # 0表示是叶子

                    clusters[i].append(new_cluster)
                    sorted_tree_edges.pop(0)
                    stop_outer_loop = True
                    break  # 倒数第三个循环停止

            if stop_outer_loop:
                break  # 倒数第四个循环停止

        if stop_outer_loop == False:
            new_cluster.append(2)  # 添加类的起始聚类
            clusters.append([new_cluster])
            sorted_tree_edges.pop(0)

    # 画出层次聚类的图
    if num:
        num += 1
        plt.figure(num)
        x = 0
        node_list = []  # 用于存储层次聚类聚类后的节点列表
        for edge in clusters[0]:
            if edge[3] == 2 and x == 0:  # 初始化，画出第一个边的两个点
                ymin = 0
                ymax = edge[2]
                plt.vlines(x, ymin=ymin, ymax=ymax, colors='blue')
                x += 1
                plt.vlines(x, ymin=ymin, ymax=ymax, colors='blue')
                plt.hlines(y=ymax, xmin=x - 1, xmax=x, color='b')
                node_list.append([x - 0.5, ymax])  # 存储节点
            elif edge[3] == 0 and ymax <= edge[2]:  # 画出一个层次的类，满足叶节点，同时边的长度逐渐增大
                ymin = ymax
                ymax = edge[2]
                node = node_list.pop()  # 弹出上一个类的节点
                plt.vlines(node[0], ymin=node[1], ymax=ymax, colors='blue')
                x += 1
                plt.vlines(x, ymin=0, ymax=ymax, colors='blue')
                plt.hlines(y=ymax, xmin=node[0], xmax=x, color='b')
                node_list.append([x - 0.5, ymax])  # 存储类的节点

            elif edge[3] == 2:  # 满足叶节点，同时边的长度没有前一个的长，则视为另一个层次的类
                ymin = 0
                ymax = edge[2]
                x += 1
                plt.vlines(x, ymin=ymin, ymax=ymax, colors='blue')
                x += 1
                plt.vlines(x, ymin=ymin, ymax=ymax, colors='blue')
                plt.hlines(y=ymax, xmin=x - 1, xmax=x, color='b')
                node_list.append([x - 0.5, ymax])

            elif edge[3] == 1:
                if len(node_list) == 1:
                    break
                node2 = node_list.pop()  # 弹出前一个类的节点
                node1 = node_list.pop()  # 弹出前前一个类的节点
                ymax = edge[2]
                plt.vlines(node1[0], ymin=node1[1], ymax=ymax, colors='blue')  # 画出上个类的节点
                plt.vlines(node2[0], ymin=node2[1], ymax=ymax, colors='blue')  # 画出上个类的节点
                plt.hlines(y=ymax, xmin=node1[0], xmax=node2[0], color='blue')
                node_list.append([(node1[0] + node2[0]) / 2, ymax])

    # 使用HDBSCAN的思想得到信号光子
    tree = clusters[0].copy()  #
    tree = tree[::-1]
    calculate_Divisive(tree)  # 计算分裂值

    # 去除散点后，得到的聚类
    cluster_node = []
    for edge in tree:
        if edge[-1] == -1:
            continue
        else:
            node1 = edge[0]
            node2 = edge[1]
            cluster_node.append(node1)
            cluster_node.append(node2)
    cluster_node = set(cluster_node)
    cluster_node = list(cluster_node)
    cluster_X = [X[index] for index in cluster_node]
    cluster_X = np.array(cluster_X)
    return cluster_X


def water_denosing(path, save_path, size):
    """
    去除水上光子
    :param path: csv文件的路径
    :param save_path: 保存文件的路径
    :param size: 选择生成UWG的方式
    :return:
    """
    data = read_csv(path)  # 读取数据

    if len(data) < 10:
        return
    # 质量分数判断
    core = Data_Quality(data['heights'].values)
    if core < 4:
        print("数据质量太差没有足够多的信号光子")
    else:
        print(f"数据分数{core}")
        min_x = data['x'].min()  # 最小值
        data['x'] = data['x'] - min_x

        min_x = data['x'].min()  # 最小值
        max_x = data['x'].max()  # 最大值

        segment_x = min_x + 100  # 每一段选取100
        segment_data = data[data.x > min_x - 20]
        segment_data = segment_data[segment_data.x < segment_x + 20]
        x = segment_data['x'].values  # numpy数组
        heights = segment_data['heights'].values  # numpy数组

        singnal_photon = HDBSCAN_my(x, heights, 4, num=False)

        with tqdm(total=max_x - segment_x, desc="Processing segments") as pbar:
            while segment_x < max_x:  # 分段对光子进行去噪
                min_x = segment_x
                segment_x = min_x + 20
                segment_data = data[data.x > min_x - 5]
                segment_data = segment_data[segment_data.x < segment_x + 5]
                x = segment_data['x'].values  # numpy数组
                heights = segment_data['heights'].values  # numpy数组
                if len(segment_data) <= 5:
                    pbar.update(20)  # 更新进度条
                    continue
                segment_singnal_photon = HDBSCAN_my(x, heights, size, num=False)
                if singnal_photon.size != 0:
                    if segment_singnal_photon.size != 0:
                        singnal_photon = np.vstack([singnal_photon, segment_singnal_photon])
                else:
                    singnal_photon = segment_singnal_photon
                pbar.update(20)  # 更新进度条

        if len(singnal_photon) >=10:
            """将信号光子数据保存在Excel里"""
            print(f"存储:{save_path}")
            column_names = ["x", "heights"]
            np.savetxt(save_path, singnal_photon, delimiter=',', fmt='%.6f', header=','.join(column_names), comments='')

            df03 = pd.DataFrame(singnal_photon, columns=column_names)
            height = calculate_height(df03)
            if height:
                return height + 1.717 - 0.32  # 高程差修正
            else:
                return None
        else:
            print("没有信号光子")
            return None




all_dir = "D:/研究生/学校/大论文/纳木错流域湖泊数据/未命名2/未命名2水位"   # 水库总文件
# 从路径中提取文件夹名称
folder_name = os.path.basename(all_dir)
print(folder_name)
# 创建CSV文件路径
csv_path = os.path.join(all_dir, f"{folder_name}.csv")
# 检查文件是否存在
if not os.path.exists(csv_path):
    # 创建一个空的数据框，并指定表头
    df = pd.DataFrame(columns=["date", "height"])

    # 将数据框保存为CSV文件
    df.to_csv(csv_path, index=False)

    print(f"CSV文件已创建: {csv_path}")
else:
    print(f"CSV文件已存在: {csv_path}")



ATL03_DIR = all_dir + "/" + "水上光子"
# 获取所有文件夹
folders = [name for name in os.listdir(ATL03_DIR) if os.path.isdir(os.path.join(ATL03_DIR, name))]
# 输出所有文件夹路径
for folder in folders:
    path = os.path.join(ATL03_DIR, folder)
    # 提取路径中的日期部分
    date = os.path.basename(path)
    print(f"开始计算水位高程：{date}")
    all_height = np.array([])
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)
            save_path = os.path.join(path, "水上信号光子")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, os.path.basename(file_path))
            print(f"保存的信号光子位置{save_path}")
            print(f"开始计算信号光子:{file_path}")
            height = water_denosing(file_path, save_path, 4)
            if height:
                all_height = np.append(all_height, height)
    if len(all_height) == 0:
        print("没有高程信息")
    else:
        print(f"水位高程列表为: {all_height}")
        water_height = np.mean(all_height)
        print(f"水位高程为: {water_height}")
        # 创建一个数据框并添加数据
        df = pd.DataFrame([[date, water_height]], columns=["date", "height"])
        df.to_csv(csv_path, mode='a', header=False, index=False)

"""
path = "D:/研究生/学校/大论文/丹江口水库/2019.04.13"
for file in os.listdir(path):
    if file.endswith('.csv'):
        file_path = os.path.join(path, file)
        save_path = os.path.join(path, "水上信号光子")
        print(file_path)
        print(save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, file)
        water_denosing(file_path, save_path, 4)
"""

