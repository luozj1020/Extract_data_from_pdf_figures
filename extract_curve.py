import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from extract_text import most_common_element
import itertools
from tqdm import tqdm
import os

from extract_image import clear_and_recreate_directory


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def kmeans_record(record):
    # 将 record 转换为列向量
    record = np.array(record).reshape(-1, 1)

    # 聚类
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(record)

    # 获取聚类结果
    labels = kmeans.labels_

    # 分离簇
    cluster_0 = record[labels == 0].flatten()  # 使用 .flatten() 确保是一维数组
    cluster_1 = record[labels == 1].flatten()

    # 获取簇的最小值和最大值
    if cluster_0.size > 0 and cluster_1.size > 0:
        max_value_cluster_0 = cluster_0.max()
        min_value_cluster_1 = cluster_1.min()

        # 判断哪个簇是较小的簇
        if max_value_cluster_0 < min_value_cluster_1:
            # 簇 0 较小，取最大值
            result_max = cluster_0.max()
            result_min = cluster_1.min()
        else:
            # 簇 1 较小，取最大值
            result_max = cluster_1.max()
            result_min = cluster_0.min()
    else:
        result_max = None
        result_min = None

    # 输出结果
    # print("取自较小簇的最大值:", result_max)
    # print("取自较大簇的最小值:", result_min)

    # 返回截图的行和列
    return result_max, result_min


def remove_axis(image_path, visualize=False):
    # 读取图像
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 逐行检测
    rows_to_record = [0, height]
    for i in range(1, height):
        if np.sum(binary[i, :] > 0) > width * 0.75:
            rows_to_record.append(i)

    # 逐列检测
    cols_to_record = [0, width]
    for j in range(1, width):
        if np.sum(binary[:, j] > 0) > height * 0.5:
            cols_to_record.append(j)

    # 输出记录的行和列
    # print("记录的行:", rows_to_record)
    # print("记录的列:", cols_to_record)

    # 调用聚类函数
    kmeans_record(rows_to_record)
    kmeans_record(cols_to_record)

    # 调用聚类函数并获取结果
    max_row, min_row = kmeans_record(rows_to_record)
    max_col, min_col = kmeans_record(cols_to_record)

    # 截图区域
    if max_row is not None and min_row is not None and max_col is not None and min_col is not None:
        # 使用最大和最小索引定义截图区域
        cropped_image = image[max_row + 1:min_row, max_col + 1:min_col]  # 包含起始索引，不包含结束索引

        if visualize:
            # 显示截图
            cv2.imshow('Cropped Image', cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return cropped_image, max_row, max_col
    else:
        print("无法提取截图，某些索引为空。")
        return image


def remove_symbols_keep_curves(image, visualize=False, lower_green=(40, 40, 40), upper_green=(80, 255, 255), min_contour_area=1000):
    """
    去除图像中的符号和文字，仅保留绿色曲线部分，并提取绿色像素的位置。

    参数:
    - image: 输入的OpenCV图像
    - lower_green: HSV下界，用于掩码绿色部分
    - upper_green: HSV上界，用于掩码绿色部分
    - min_contour_area: 轮廓面积最小阈值，小于此面积的轮廓将被去除

    返回:
    - result: 仅保留曲线部分的图像 (OpenCV图像格式)
    - green_positions: 绿色像素的坐标位置列表 [(x, y), ...]
    """
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建绿色掩码
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 找到所有轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个全黑的掩码来保留曲线
    curve_mask = np.zeros_like(mask)

    # 过滤掉较小的轮廓，仅保留曲线
    for cnt in contours:
        if cv2.contourArea(cnt) > min_contour_area:
            # 填充较大的轮廓以保留曲线
            cv2.drawContours(curve_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # 使用曲线掩码过滤原始图像，仅保留曲线部分
    result = cv2.bitwise_and(image, image, mask=curve_mask)

    # 提取绿色像素的位置
    green_positions = []
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            if np.array_equal(result[y, x], [0, 255, 0]):  # 检查是否为绿色像素
                green_positions.append((x, y))

    if visualize:
        cv2.imshow("Filtered Image", result)
        cv2.imwrite("filtered_curve_only.png", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return green_positions


def visualize_pixel_blocks(binary_image, pixel_blocks):
    # 创建一个与输入图像大小一致的显示图像
    height, width = binary_image.shape
    visualized_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 颜色映射表，生成足够的颜色以区分块
    np.random.seed(0)  # 固定随机种子以便复现
    colors = np.random.randint(0, 255, (len(pixel_blocks), 3))

    # 将每个块绘制到图像中
    for block_id, points in pixel_blocks.items():
        color = colors[block_id - 1]
        for x, y in points:
            visualized_image[x, y] = color

    # 可视化
    plt.figure(figsize=(8, 8))
    plt.imshow(visualized_image)
    plt.axis("off")
    plt.title("Pixel Blocks Visualization")
    plt.show()


def find_pixel_blocks(binary_image, visualize=False):
    height, width = binary_image.shape
    visited = np.zeros_like(binary_image, dtype=bool)
    pixel_blocks = {}

    def bfs(start_x, start_y):
        # BFS 初始化
        queue = [(start_x, start_y)]
        visited[start_x, start_y] = True  # 在入队时就标记为已访问
        pixel_points = []

        while queue:
            x, y = queue.pop(0)
            pixel_points.append((x, y))

            # 检查上下左右的邻居像素
            for dx, dy in [(-1,0),(1,0),(0,1),(0,-1)]:
                nx, ny = x + dx, y + dy
                # 只有符合条件的邻居才入队
                if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny] and binary_image[nx, ny] > 0:
                    queue.append((nx, ny))
                    visited[nx, ny] = True  # 入队时标记为已访问

        return pixel_points

    # 遍历整个图像，找到所有连通像素块
    block_id = 0
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] > 0 and not visited[i, j]:
                block_id += 1
                pixel_points = bfs(i, j)
                pixel_blocks[block_id] = pixel_points

    if visualize:
        visualize_pixel_blocks(binary_image, pixel_blocks)

    return pixel_blocks


def kmeans_pixel_blocks(cropped_image, threshold=1000):
    # 假设 cropped_image 已经提取
    gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, binary_cropped = cv2.threshold(gray_cropped, 240, 255, cv2.THRESH_BINARY_INV)

    # 查找像素块
    pixel_blocks = find_pixel_blocks(binary_cropped)

    # 计算每个像素块的像素数量
    pixel_counts = np.array([len(points) for points in pixel_blocks.values()]).reshape(-1, 1)

    # 计算新的 pixel_counts 值：10000 * tanh(x - threshold)
    adjusted_pixel_counts = 10000 * np.tanh(pixel_counts - threshold)

    # 进行 K-means 聚类
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(adjusted_pixel_counts)

    # 获取聚类标签
    labels = kmeans.labels_

    # 将像素块按聚类结果分组
    clusters = {0: [], 1: []}
    for block_id, label in zip(pixel_blocks.keys(), labels):
        clusters[label].append(block_id)

    # 计算每个簇的总像素数量
    cluster_sums = {0: sum(len(pixel_blocks[b]) for b in clusters[0]),
                    1: sum(len(pixel_blocks[b]) for b in clusters[1])}

    # 找到像素数量较多的簇
    larger_cluster_label = 0 if cluster_sums[0] >= cluster_sums[1] else 1
    selected_blocks = clusters[larger_cluster_label]

    # 输出保留的像素块
    print(f"保留的像素块编号 (数量较多的簇): {selected_blocks}")

    # 可视化保留的像素块
    result_image = np.zeros_like(cropped_image)
    for block_id in selected_blocks:
        for (x, y) in pixel_blocks[block_id]:
            result_image[x, y] = (0, 255, 0)  # 使用绿色标记保留的像素块

    selected_pixels = remove_symbols_keep_curves(result_image)

    return selected_pixels


def filter_pixels(pixel_list):
    # 将像素坐标转为集合，方便快速查找
    pixel_set = set(tuple(pixel) for pixel in pixel_list)

    # 用来存储保留的像素
    remaining_pixels = []

    # 遍历每个像素
    for (x, y) in pixel_list:
        # 计算四个方向的邻居坐标
        neighbors = [
            (x, y - 1),  # 上
            (x, y + 1),  # 下
            (x - 1, y),  # 左
            (x + 1, y),  # 右
        ]

        # 计算有多少个邻居在集合中
        neighbor_count = sum(1 for neighbor in neighbors if neighbor in pixel_set)

        # 如果至少有两个邻居存在，保留该像素
        if neighbor_count >= 2:
            remaining_pixels.append((x, y))

    return remaining_pixels


def find_pixel_blocks_from_list(pixel_list):
    # 创建一个用于标记已访问像素的集合
    visited = set()
    pixel_blocks = {}

    # 将像素列表转化为集合，方便邻域检查
    pixel_set = set(pixel_list)

    def bfs(start_x, start_y):
        queue = deque([(start_x, start_y)])
        pixel_points = []

        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            pixel_points.append((x, y))

            # 仅检查上下左右的像素，且这些像素也在 pixel_list 中
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in pixel_set and (nx, ny) not in visited:
                        # 仅添加相邻的像素（这些像素是当前块的一部分）
                        queue.append((nx, ny))

        return pixel_points

    block_id = 0
    for (x, y) in pixel_list:
        if (x, y) not in visited:
            block_id += 1
            pixel_points = bfs(x, y)
            pixel_blocks[block_id] = filter_pixels(pixel_points)

    pixel_blocks_ = {}
    for block_id, pixel_block in pixel_blocks.items():
        if len(pixel_block) > 10:
            pixel_blocks_[block_id] = pixel_block

    return pixel_blocks_


def kmeans_pixel_blocks_from_list(pixel_list):
    """
    该函数将像素分成块，并在这些块上执行 MiniBatch KMeans 聚类，
    从而识别出最重要的块（基于像素数量）。

    参数:
    - pixel_list: 像素坐标（元组的列表）

    返回:
    - result_pixels: 聚类后选中的像素列表
    """
    # 第一步：查找像素块
    pixel_blocks = find_pixel_blocks_from_list(pixel_list)  # 确保此函数能正确分组像素

    if len(pixel_blocks.keys()) > 2:
        # 第二步：计算每个像素块的像素数量
        pixel_counts = np.array([len(points) for points in pixel_blocks.values()]).reshape(-1, 1)
        print("pixel_counts shape:", pixel_counts.shape)  # 检查 pixel_counts 的形状

        # 如果只有一个像素块，直接返回该块
        if pixel_counts.shape[0] == 1:
            result_pixels = list(pixel_blocks.values())[0]  # 只有一个块，直接返回
            return result_pixels

        # 第三步：对像素块的数量使用 MiniBatchKMeans 聚类
        kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=200).fit(pixel_counts)

        # 第四步：获取聚类标签
        labels = kmeans.labels_

        # 第五步：根据聚类标签将像素块分组
        clusters = {0: [], 1: []}
        for block_id, label in zip(pixel_blocks.keys(), labels):
            clusters[label].append(block_id)

        # 第六步：计算每个簇的总像素数量
        cluster_sums = {
            0: sum(len(pixel_blocks[b]) for b in clusters[0]),
            1: sum(len(pixel_blocks[b]) for b in clusters[1])
        }

        # 第七步：选择像素数量较多的簇
        larger_cluster_label = 0 if cluster_sums[0] >= cluster_sums[1] else 1
        selected_blocks = clusters[larger_cluster_label]

        # 第八步：合并所选块中的像素
        result_pixels = set()
        for block_id in selected_blocks:
            result_pixels.update(pixel_blocks[block_id])  # 合并所选块中的像素

    else:
        result_pixels = set()
        for block_id in list(pixel_blocks.keys()):
            result_pixels.update(pixel_blocks[block_id])  # 合并所选块中的像素

    return list(result_pixels)  # 返回选中的像素坐标列表


def group_pixels_by_x_and_y(selected_pixels):
    x_dict = defaultdict(list)

    # 按 x 坐标分组
    for x, y in selected_pixels:
        x_dict[x].append(y)

    # 对每个 x 坐标相同的像素组，按 y 坐标分组连续的像素块
    for x in x_dict:
        y_values = sorted(x_dict[x])  # 对 y 坐标排序
        pixel_blocks = []  # 存储连续 y 坐标的像素块
        block = [y_values[0]]  # 初始化第一个块

        # 遍历 y 坐标，找到连续的像素块
        for i in range(1, len(y_values)):
            if y_values[i] == y_values[i - 1] + 1:
                block.append(y_values[i])  # 连续的 y 值加入同一个块
            else:
                pixel_blocks.append(block)  # 结束当前块，加入块列表
                block = [y_values[i]]  # 开始一个新的块

        pixel_blocks.append(block)  # 添加最后的块
        x_dict[x] = pixel_blocks  # 将像素块的列表存回 x_dict 中

    return x_dict


def store_curve_colors(grouped_pixels, cropped_image, curve_num):
    """
    Store the RGB colors of each curve in a dictionary, ensuring unique colors per curve,
    and store the corresponding pixel coordinates.

    参数:
    - grouped_pixels: 包含每条曲线的像素块列表的字典
    - cropped_image: 处理后的图像，用于获取像素颜色
    - curve_num: 曲线的数量

    返回:
    - curve_colors_dict: 字典，键为曲线编号，值为存储该曲线RGB颜色的唯一列表
    - curve_pixels_dict: 字典，键为曲线编号，值为存储该曲线所有匹配的像素坐标列表
    """
    # 创建字典来存储每条曲线的RGB颜色和像素坐标
    curve_colors_dict = {i: [] for i in range(curve_num)}
    curve_pixels_dict = {i: [] for i in range(curve_num)}

    # 遍历grouped_pixels中的像素块
    n = 10
    check = True
    start = 0
    for x, pixel_blocks in grouped_pixels.items():
        if len(pixel_blocks) == curve_num:  # 判断该曲线的像素块数量是否与curve_num相等
            check = False
            for i in range(1, n + 1):
                if list(grouped_pixels.keys()).index(x) + i < len(grouped_pixels.items()):
                    if len(list(grouped_pixels.items())[list(grouped_pixels.keys()).index(x) + i][1]) != curve_num:
                        check = True
                        break
            if not check:
                for sub_block in pixel_blocks:  # 遍历每个子块（像素位置）
                    for y in sub_block:  # 获取子块中的每个像素位置
                        color = cropped_image[y, x]  # 获取该像素的RGB颜色
                        # 使用sub_block的索引来确定曲线对应的颜色
                        curve_index = pixel_blocks.index(sub_block)  # 获取曲线的索引
                        curve_colors_dict[curve_index].append(tuple(color))  # 保证颜色是元组形式，避免引用问题
                        curve_pixels_dict[curve_index].append((x, y))  # 保存该像素的位置
        else:
            if check:
                pass
            else:
                break
        start += 1

    check = True
    end = len(grouped_pixels.items()) - 1
    for x, pixel_blocks in list(grouped_pixels.items())[::-1]:
        if len(pixel_blocks) == curve_num:  # 判断该曲线的像素块数量是否与curve_num相等
            check = False
            for i in range(1, n + 1):
                if len(list(grouped_pixels.items())[list(grouped_pixels.keys()).index(x) - i][1]) != curve_num:
                    check = True
                    break
            if not check:
                for sub_block in pixel_blocks:  # 遍历每个子块（像素位置）
                    for y in sub_block:  # 获取子块中的每个像素位置
                        color = cropped_image[y, x]  # 获取该像素的RGB颜色
                        # 使用sub_block的索引来确定曲线对应的颜色
                        curve_index = pixel_blocks.index(sub_block)  # 获取曲线的索引
                        curve_colors_dict[curve_index].append(tuple(color))  # 保证颜色是元组形式，避免引用问题
                        curve_pixels_dict[curve_index].append((x, y))  # 保存该像素的位置
        else:
            if check:
                pass
            else:
                break
        end -= 1
        if end == start:
            break

    # 检查是否存在灰色（RGB值相等的颜色）
    contains_gray = any(
        any(color[0] == color[1] == color[2] for color in colors)
        for colors in curve_colors_dict.values()
    )

    if not contains_gray:
        for curve_id in list(curve_colors_dict.keys()):
            if (0, 0, 0) in curve_colors_dict[curve_id]:
                # 移除黑色颜色和对应的像素
                black_indices = [
                    i for i, color in enumerate(curve_colors_dict[curve_id]) if color == (0, 0, 0)
                ]
                curve_colors_dict[curve_id] = [
                    color for color in curve_colors_dict[curve_id] if color != (0, 0, 0)
                ]
                curve_pixels_dict[curve_id] = [
                    pixel for i, pixel in enumerate(curve_pixels_dict[curve_id])
                    if i not in black_indices
                ]

    # 对每条曲线的颜色进行去重
    for curve_id in curve_colors_dict:
        curve_colors_dict[curve_id] = list(set(curve_colors_dict[curve_id]))
        curve_pixels_dict[curve_id] = list(set(curve_pixels_dict[curve_id]))

    return curve_colors_dict, curve_pixels_dict


def find_curve_pixels_dict(cropped_image, curve_colors_dict, curve_pixels_dict, selected_pixels):
    """
    Find pixels in cropped_image that match the colors of each curve and store their coordinates.
    This function now simply adds pixels to the corresponding curve in curve_pixels_dict.

    参数:
    - cropped_image: 处理后的图像，用于获取像素颜色
    - curve_colors_dict: 每条曲线的RGB颜色列表字典
    - curve_pixels_dict: 存储曲线像素坐标的字典

    返回:
    - curve_pixels_dict: 更新后的字典，包含每条曲线的匹配像素坐标
    """
    height, width, _ = cropped_image.shape

    # 1. Convert curve colors to sets of tuples for fast lookup
    curve_colors_sets = {
        curve_id: set(tuple(color) for color in colors)  # Convert each color to a tuple and store in a set
        for curve_id, colors in curve_colors_dict.items()
    }

    # 2. Create a set of already processed pixels (merged_list)
    merged_list = set(itertools.chain(*curve_pixels_dict.values()))  # Set for fast lookup

    # 3. Iterate through selected_pixels
    for (x, y) in tqdm(selected_pixels):
        if (x, y) not in merged_list:
            pixel_color = tuple(cropped_image[y, x])  # Convert the color to a tuple for comparison

            # 4. Check if the pixel matches any curve color
            for curve_id, color_set in curve_colors_sets.items():
                if pixel_color in color_set:  # Fast lookup in set
                    curve_pixels_dict[curve_id].append((x, y))  # Add the pixel to the corresponding curve
                    # break  # Once matched, no need to check other curves

    for curve_id in list(curve_pixels_dict.keys()):
        curve_pixels_dict[curve_id] = kmeans_pixel_blocks_from_list(curve_pixels_dict[curve_id])

    return curve_pixels_dict


def save_curve_pixels_dict_separately(cropped_image, curve_pixels_dict, curve_num, image_path, output_dir):
    """
    Save the pixels of each curve in separate image files.

    参数:
    - cropped_image: 处理后的图像，用于展示曲线
    - curve_pixels_dict: 每条曲线的像素坐标字典
    - curve_num: 曲线的数量，决定每条曲线使用不同颜色
    - output_dir: 保存图像的目录路径

    返回:
    - None: 每条曲线的图像将被保存为单独的文件
    """

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 为每条曲线分配一个颜色
    colors = plt.cm.get_cmap('tab10', curve_num)  # 使用 Matplotlib 的 colormap 来生成不同颜色

    # 遍历每条曲线
    for curve_id in range(curve_num):
        # 创建一个空白图像，初始化为黑色
        visualized_image = np.zeros_like(cropped_image)

        # 获取当前曲线的像素
        pixels = curve_pixels_dict[curve_id]

        # 为该曲线分配颜色
        color = colors(curve_id)[:3]  # 获取颜色并丢弃 alpha 通道
        color = tuple(int(c * 255) for c in color)  # 转换为 RGB (0-255)

        # 将该曲线的像素点绘制到图像中
        for (x, y) in pixels:
            visualized_image[y, x] = color  # 使用 curve_id 的颜色绘制曲线像素

        # 保存每条曲线的图像
        output_path = f"{output_dir}/{os.path.splitext(os.path.basename(image_path))[0]}_curve_{curve_id + 1}_pixels.png"
        cv2.imwrite(output_path, visualized_image)
        print(f"Saved Curve {curve_id + 1} Image at {output_path}")


def linear_interpolate(points):
    """
    对缺失的点进行线性插值，填补数据空缺。

    参数:
    points: List[Tuple[int, int]] - 点的列表，形式为 [(x, y)]。

    返回:
    List[Tuple[int, int]] - 插值后的点列表。
    """
    # 将点按 x 排序
    points = sorted(points, key=lambda p: p[0])

    # 处理只有左侧或右侧的点时的插值
    interpolated_points = []
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]

        # 如果x值相差不大，可以插值
        if x2 > x1 + 1:
            for x in range(x1 + 1, x2):
                y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                interpolated_points.append((x, y))

        # 保留原始点
        interpolated_points.append((x1, y1))

    # 添加最后一个点
    interpolated_points.append(points[-1])

    return interpolated_points


def calculate_local_slope(points, window_size=3):
    """
    使用局部线性回归来计算点的局部斜率。

    参数:
    points: List[Tuple[int, int]] - 点的列表，形式为 [(x, y)]。
    window_size: int - 滑动窗口的大小，控制局部区域的范围。

    返回:
    slopes: dict - 每个x值对应的局部斜率。
    """
    # 根据 x 值对点进行排序
    points = sorted(points, key=lambda p: p[0])

    slopes = {}
    for i in range(len(points)):
        # 提取局部窗口
        start = max(0, i - window_size // 2)
        end = min(len(points), i + window_size // 2 + 1)
        local_points = points[start:end]

        # 提取x和y值
        x_vals = np.array([p[0] for p in local_points])
        y_vals = np.array([p[1] for p in local_points])

        # 进行局部线性回归，计算斜率
        if len(x_vals) > 1:
            A = np.vstack([x_vals, np.ones(len(x_vals))]).T
            slope, _ = np.linalg.lstsq(A, y_vals, rcond=None)[0]
            slopes[points[i][0]] = slope

    return slopes


def smooth_and_fill_from_points(points):
    # 1. 根据输入的像素点计算图像的大小
    if not points:
        return []  # 如果输入为空，返回空列表

    # 获取最大最小的 x 和 y 坐标
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    # 根据最大最小值设置图像大小，并扩展一些空间以避免边界问题
    image_shape = (max_y - min_y + 10, max_x - min_x + 10)  # 留出一些边距

    # 2. 创建一个空白图像
    image = np.zeros(image_shape, dtype=np.uint8)

    # 3. 将输入的像素点列表绘制到图像上（白色表示曲线）
    for (x, y) in points:
        # 将坐标平移到图像的坐标系中
        image[y - min_y, x - min_x] = 255

    # 4. 使用高斯滤波对图像进行平滑
    smoothed_image = gaussian_filter(image.astype(float), sigma=1)

    # 5. 二值化处理：将平滑后的图像转换为二值图像
    _, binary_image = cv2.threshold(smoothed_image, 127, 255, cv2.THRESH_BINARY)

    # 6. 使用形态学操作（膨胀和腐蚀）来去除小毛刺并填充沟壑
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

    # 7. 提取修复后的像素点坐标
    points_after_smoothing = np.column_stack(np.where(eroded_image == 255))

    # 将坐标转换回原始坐标系
    output_points = [(x + min_x, y + min_y) for y, x in points_after_smoothing]  # 转换回 (x, y) 顺序

    # 返回修复后的像素点列表
    return output_points


def filter_points_by_slope_difference(points, min_height, max_height, loss_x, slope_threshold=0.01, window_size=3):
    """
    根据局部线性回归斜率差来过滤曲线较细部分的像素点。

    参数:
    points: List[Tuple[int, int]] - 已知的曲线点，形式为 [(x, y)]。
    slope_threshold: float - 斜率差异阈值，超过此差异的x值对应点会被去掉。
    window_size: int - 局部线性回归的滑动窗口大小。

    返回:
    List[Tuple[int, int]] - 过滤后的曲线点。
    """
    x_to_remove = set()
    points = smooth_and_fill_from_points(points)
    min_x = min([p[0] for p in points])
    max_x = max([p[0] for p in points])
    # Step 1: 获取每个x值对应的y的最大值和最小值
    x_groups = defaultdict(list)
    for x, y in points:
        x_groups[x].append(y)

    max_points = [(x, max(y_list)) for x, y_list in x_groups.items()]
    min_points = [(x, min(y_list)) for x, y_list in x_groups.items()]

    # Step 2: 计算 max_points 和 min_points 的局部线性回归斜率
    max_slopes = calculate_local_slope(max_points, window_size)
    min_slopes = calculate_local_slope(min_points, window_size)

    for x, y_list in x_groups.items():
        if (max(y_list) - min(y_list)) <= min_height or (max(y_list) - min(y_list)) >= max_height:
            x_to_remove.add(x)

    max_points_dict = defaultdict(list)
    for x, y in max_points:
        max_points_dict[x].append(y)
    min_points_dict = defaultdict(list)
    for x, y in min_points:
        min_points_dict[x].append(y)
    # Step 3: 筛选出斜率差异大于阈值的x值
    for x in max_slopes:
        if x in loss_x:
            if x in min_slopes:
                slope_difference = abs(max_slopes[x] - min_slopes[x])
                if min_x < x < max_x:
                    if slope_difference > slope_threshold:
                        if abs(max_slopes[x]) < slope_threshold and abs(min_slopes[x]) < slope_threshold:
                            pass
                        else:
                            x_to_remove.add(x)

    # Step 4: 在原始 points 中去掉这些x值对应的所有点
    filtered_points = [(x, y) for x, y in points if x not in x_to_remove]

    return filtered_points


def find_and_expand_pixels(pixels, selected_pixels, mean_height):
    """
    扩展初始像素列表 `pixels`，包括补全缺失像素和递归查找前向像素。

    参数:
        pixels (list of tuple): 初始像素列表，每个元素为 (x, y)。
        selected_pixels (list of tuple): 用于补全和扩展的参考像素列表。
        mean_height (int): 平均高度，用于动态扩展相邻像素的条件。

    返回:
        list of tuple: 所有扩展后的像素点，按 (x, y) 排序。
    """
    # 将 pixels 转换为集合，方便查找和去重
    pixels_set = set(pixels)
    selected_pixels_set = set(selected_pixels)

    # 创建一个字典来存储按 x 分类的 y 值列表
    pixel_dict = defaultdict(list)
    for x, y in pixels:
        pixel_dict[x].append(y)

    # 对键排序并更新原字典
    sorted_items = sorted(pixel_dict.items())  # 按键排序
    pixel_dict = defaultdict(list, sorted_items)  # 重新赋值为排序后的字典

    # 补全 pixels 中的缺失相邻像素
    min_x = min(p[0] for p in pixels)
    max_x = max(p[0] for p in pixels)
    # 向前遍历
    for x in sorted(pixel_dict.keys()):  # 遍历排序后的键
        y_list = pixel_dict[x]
        if min_x < x < max_x:  # 只处理中间范围的 x
            left_height, right_height = 0, 0
            # 计算相邻列的高度差
            if x - 1 in pixel_dict:
                left_height = max(pixel_dict[x - 1]) - min(pixel_dict[x - 1])
            if x + 1 in pixel_dict:
                right_height = max(pixel_dict[x + 1]) - min(pixel_dict[x + 1])
            # 期望的高度差
            expected_height = max(left_height, right_height)
            # 如果当前列的高度差小于期望值，进行补全
            if max(y_list) - min(y_list) < expected_height:
                y_max, y_min = max(y_list), min(y_list)
                # 扩展 y_list 直到达到期望高度
                while True:
                    extend = False
                    if (x, y_min - 1) not in pixels_set and (y_max - y_min + 1) < expected_height:
                        pixels_set.add((x, y_min - 1))
                        selected_pixels_set.add((x, y_min - 1))
                        y_list.append(y_min - 1)
                        y_min -= 1
                        extend = True
                    if (x, y_max + 1) not in pixels_set and (y_max - y_min + 1) < expected_height:
                        pixels_set.add((x, y_max + 1))
                        selected_pixels_set.add((x, y_max + 1))
                        y_list.append(y_max + 1)
                        y_max += 1
                        extend = True
                    if not extend or (y_max - y_min + 1) >= expected_height:
                        break
                # 更新字典中的 y_list
                pixel_dict[x] = sorted(y_list)  # 排序以保持顺序性
    # 向后遍历
    for x in sorted(pixel_dict.keys())[::-1]:  # 遍历排序后的键
        y_list = pixel_dict[x]
        if min_x < x < max_x:  # 只处理中间范围的 x
            left_height, right_height = 0, 0
            # 计算相邻列的高度差
            if x - 1 in pixel_dict:
                left_height = max(pixel_dict[x - 1]) - min(pixel_dict[x - 1])
            if x + 1 in pixel_dict:
                right_height = max(pixel_dict[x + 1]) - min(pixel_dict[x + 1])
            # 期望的高度差
            expected_height = max(left_height, right_height)
            # 如果当前列的高度差小于期望值，进行补全
            if max(y_list) - min(y_list) < expected_height:
                y_max, y_min = max(y_list), min(y_list)
                # 扩展 y_list 直到达到期望高度
                while True:
                    extend = False
                    if (x, y_min - 1) not in pixels_set and (y_max - y_min + 1) < expected_height:
                        pixels_set.add((x, y_min - 1))
                        selected_pixels_set.add((x, y_min - 1))
                        y_list.append(y_min - 1)
                        y_min -= 1
                        extend = True
                    if (x, y_max + 1) not in pixels_set and (y_max - y_min + 1) < expected_height:
                        pixels_set.add((x, y_max + 1))
                        selected_pixels_set.add((x, y_max + 1))
                        y_list.append(y_max + 1)
                        y_max += 1
                        extend = True
                    if not extend or (y_max - y_min + 1) >= expected_height:
                        break
                # 更新字典中的 y_list
                pixel_dict[x] = sorted(y_list)  # 排序以保持顺序性

    # 按 x 从小到大排序字典的键
    sorted_x_keys = sorted(pixel_dict.keys())

    # 递归向前查找像素
    loss = 0
    loss_x = []
    current_x = sorted_x_keys[0]  # 从最小的 x 开始
    while True:
        next_x = current_x + 1
        if next_x > max(sorted_x_keys):
            break
        if next_x not in sorted_x_keys:
            loss += 1
            loss_x.append(next_x)
            loss_x.append(current_x)
            # 获取当前 x 的 y_min 和 y_max
            y_min = min(pixel_dict[current_x])
            y_max = max(pixel_dict[current_x])

            # 查找 selected_pixels 中满足条件的横坐标为 next_x 的像素
            new_pixels = [
                (px, py) for px, py in selected_pixels_set
                if px == next_x and y_min <= py <= y_max
            ]

            while new_pixels:
                # 当前 new_pixels 的 y 范围
                current_y_min = min(p[1] for p in new_pixels)
                current_y_max = max(p[1] for p in new_pixels)

                # 初始化一个集合用于存储新添加的像素
                additional_pixels = set()

                # 动态扩展下边界
                if current_y_max < y_max and (current_y_max - current_y_min) < mean_height:
                    additional_pixels.update(
                        (px, py + 1) for px, py in new_pixels
                        if (px, py + 1) in selected_pixels_set and (px, py + 1) not in pixels_set
                    )

                # 动态扩展上边界
                if current_y_min > y_min and (current_y_max - current_y_min) < mean_height:
                    additional_pixels.update(
                        (px, py - 1) for px, py in new_pixels
                        if (px, py - 1) in selected_pixels_set and (px, py - 1) not in pixels_set
                    )

                # 如果没有新像素添加，退出循环
                if not additional_pixels:
                    break

                # 更新 new_pixels 和 pixels_set
                new_pixels = list(additional_pixels)
                pixels_set.update(new_pixels)

                # 停止条件：扩展范围达到指定大小
                if max(current_y_max, max(p[1] for p in new_pixels)) - min(current_y_min,
                                                                           min(p[1] for p in new_pixels)) >= min(
                    mean_height, y_max - y_min):
                    break

            # 去重并排序 new_pixels
            new_pixels = list(set(new_pixels))

            # 如果没有找到新像素，则结束
            if not new_pixels:
                break

            # 更新 pixel_dict 和 pixels_set
            pixel_dict[next_x].extend(py for px, py in new_pixels if px == next_x)
            pixels_set.update(new_pixels)

            # 去重并排序 pixel_dict[next_x]
            pixel_dict[next_x] = sorted(set(pixel_dict[next_x]))

        # 移动到下一个 x
        current_x = next_x

    loss_ratio = loss / (max(sorted_x_keys) - min(sorted_x_keys))

    # 最终返回所有扩展后的像素点，按 (x, y) 排序
    return sorted(pixels_set, key=lambda p: (p[0], p[1])), loss_ratio, list(set(loss_x))


def interpolate_and_average_points(points, selected_pixels, visualize=False):
    """
    基于已有的像素点列表推测并补全缺失的x坐标对应的y值，并返回清洗后的平均像素点和补全的插值像素点。

    参数:
    points (list of tuple): 已知的部分曲线点，形式为 [(x, y)]，每个x可能对应多个y值。

    返回:
    final_points (list of tuple): 合并清洗后的平均像素点和补充的插值像素点。
    """

    # 0. 清洗数据：将曲线较细部分的数据过滤掉
    x_groups = defaultdict(list)
    for x, y in points:
        x_groups[x].append(y)
    mean_height = np.mean([(max(y_list) - min(y_list)) for x, y_list in x_groups.items()])
    min_height = min([(max(y_list) - min(y_list)) for x, y_list in x_groups.items()])
    max_height = max([(max(y_list) - min(y_list)) for x, y_list in x_groups.items()])
    points, loss_ratio, loss_x = find_and_expand_pixels(points, selected_pixels, mean_height)
    print('loss_ratio:', loss_ratio)
    if loss_ratio >= 0.01:
        points = filter_points_by_slope_difference(points, min_height, max_height, loss_x)
    else:
        points = smooth_and_fill_from_points(points)

    # 1. 整理数据：将相同的x值的y值放入一个字典中
    x_dict = defaultdict(list)
    for x, y in points:
        x_dict[x].append(y)

    # 2. 提取已知的x数据
    x_known = sorted(x_dict.keys())

    # 3. 对于每个x，取其y值的最大值和最小值
    y_max = [np.max(x_dict[x]) for x in x_known]
    y_min = [np.min(x_dict[x]) for x in x_known]

    # 4. 找出缺失的x值
    x_missing = []
    for i in range(1, len(x_known)):
        if x_known[i] - x_known[i - 1] > 1:
            missing_x_range = np.arange(x_known[i - 1] + 1, x_known[i])
            x_missing.extend(missing_x_range)

    x_missing = np.array(x_missing)

    # 5. 使用最大值和最小值进行插值
    cs_max = CubicSpline(x_known, y_max)
    cs_min = CubicSpline(x_known, y_min)

    # 6. 预测缺失点的最大值和最小值
    y_missing_max = cs_max(x_missing)
    y_missing_min = cs_min(x_missing)

    # 7. 计算缺失点的平均y值
    y_missing = (y_missing_max + y_missing_min) / 2

    # 8. 合并插值补充的点和清洗后的点
    interpolated_points = list(zip(x_missing, y_missing))
    averaged_points = [(x, np.mean(y_vals)) for x, y_vals in x_dict.items()]
    final_points = averaged_points + interpolated_points

    # 9. 可视化：绘制原始曲线、插值曲线和最终结果
    x_all = np.linspace(min(x_known), max(x_known), 200)
    y_all_max = cs_max(x_all)
    y_all_min = cs_min(x_all)

    if visualize:
        # 画出原始点、最大值插值、最小值插值和最终补全结果
        # plt.plot(x_known, -np.array(y_max), 'ro', label="最大值插值点")
        # plt.plot(x_known, -np.array(y_min), 'bo', label="最小值插值点")
        plt.plot(x_all, -y_all_max, 'r-', label="最大值插值曲线")
        plt.plot(x_all, -y_all_min, 'b-', label="最小值插值曲线")
        # plt.plot(x_missing, -y_missing, 'go', label="补全点")

        # 画出最终结果的点
        final_x, final_y = zip(*final_points)
        plt.plot(final_x, -np.array(final_y), 'mo', label="最终结果点")

        plt.legend()
        plt.xlabel("X坐标")
        plt.ylabel("Y坐标")
        plt.title("基于最大值和最小值插值进行补全后的最终结果")
        plt.show()

    return final_points


def extract_curve(image_path, output_dir, visualize=False):
    cropped_image, x_axis_upper, y_axis_left = remove_axis(image_path)
    selected_pixels = kmeans_pixel_blocks(cropped_image)
    # 获取分组后的像素块
    grouped_pixels = group_pixels_by_x_and_y(selected_pixels)

    # 准备数据以绘制直方图
    x_values = list(grouped_pixels.keys())
    block_lengths = [len(blocks) for blocks in grouped_pixels.values()]
    curve_num = most_common_element(block_lengths)  # 将 block_lengths 出现最多的长度作为曲线数量

    if visualize:
        # 绘制直方图
        plt.figure(figsize=(8, 5))
        plt.bar(x_values, block_lengths, color='skyblue')
        plt.xlabel('X Coordinate')
        plt.ylabel('Number of Pixel Blocks')
        plt.title('Histogram of Pixel Blocks per X Coordinate')
        plt.show()
        plt.savefig('Number of Pixel Blocks.png')

    curve_colors, curve_pixels_dict = store_curve_colors(grouped_pixels, cropped_image, curve_num)

    # 调用find_curve_pixels_dict来更新曲线的像素坐标
    curve_pixels_dict = find_curve_pixels_dict(cropped_image, curve_colors, curve_pixels_dict, selected_pixels)

    # 保存每条曲线的像素图像
    save_curve_pixels_dict_separately(cropped_image, curve_pixels_dict, curve_num, image_path, output_dir)

    # 处理缺失数据
    # selected_pixels = list(itertools.chain(*curve_pixels_dict.values()))
    final_points_dict = {}
    for curve_id, pixels in curve_pixels_dict.items():
        print(curve_id)
        final_points = interpolate_and_average_points(pixels, selected_pixels)
        final_points_dict[curve_id] = [(p[0]+x_axis_upper, p[1]+y_axis_left) for p in final_points]

    return final_points_dict, curve_colors


if __name__ == "__main__":
    image_path = 'example_fig5_part_1_part_2.png'
    output_dir = 'output_images'  # 设置保存图像的目录
    clear_and_recreate_directory(output_dir)
    final_points_dict, curve_colors = extract_curve(image_path, output_dir, visualize=False)

