from collections import Counter
import pytesseract
from PIL import Image
import pandas as pd
from paddleocr import PaddleOCR
from itertools import combinations

from extract_image import ocr_image
from closest_bbox import map_curve_to_bbox
from extract_curve import *
from string_operation import *


def extract_color_position_dict(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    # 转换为 RGB 格式
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape

    # 初始化结果字典
    color_position_dict = {}

    for y in range(height):
        for x in range(width):
            # 获取像素的 RGB 值
            r, g, b = img[y, x]

            # 忽略白色、灰色和黑色像素
            if r == g == b:
                continue

            # 将颜色元组作为键
            color = (r, g, b)
            if color not in color_position_dict:
                color_position_dict[color] = []

            # 添加像素位置到列表
            color_position_dict[color].append((x, y))

    return color_position_dict


def filter_points_and_related(points, related, threshold):
    # 将列表转换为 numpy 数组
    points_array = np.array(points)
    related_array = np.array(related)

    # 提取 y 坐标
    y_coords = points_array[:, 1]

    # 筛选相近的 y 坐标及相关的元素
    filtered_points = []
    filtered_related = []

    for i, point in enumerate(points_array):
        # 检查当前点的 y 坐标是否与其他点相近
        if any(np.abs(y_coords[i] - y_coords[j]) <= threshold for j in range(len(y_coords)) if j != i):
            filtered_points.append(tuple(point))
            filtered_related.append(related_array[i])

    return filtered_points, filtered_related


def extract_axis_values(image_path, custom_config):
    # 加载图像
    image = Image.open(image_path)
    data = ocr_image(image_path, custom_config)
    print('ocr_image data:', data['text'])

    x_values = []
    x_positions = []

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text.replace('.', '', 1).isdigit():  # 检查是否为数字
            # 获取每个识别结果的坐标和文本
            x = data['top'][i] + data['height'][i]/2

            # 假设 x 轴的坐标位于图像底部，y 轴的坐标位于图像左侧
            if x > image.height * 0.7:  # 靠底部，假设是 x 轴
                x_values.append(text)
                x_positions.append((data['left'][i] + data['width'][i]/2, data['top'][i] + data['height'][i]/2))
    # print(x_positions, x_values)
    x_positions, x_values = filter_points_and_related(x_positions, x_values, 10)
    x_values = np.array(x_values, dtype=float)
    x_positions = np.array(x_positions, dtype=float)

    return x_values, x_positions


def get_max_and_min_y(final_points_dict):
    all_positions = [pos for positions in final_points_dict.values() for pos in positions]

    max_y_position = max(all_positions, key=lambda p: p[1])
    min_y_position = min(all_positions, key=lambda p: p[1])

    return max_y_position[1], min_y_position[1]


def fix_arithmetic_sequence(sequence):
    # 计算相邻差值
    diffs = [sequence[i] - sequence[i - 1] for i in range(1, len(sequence))]

    # 找到最常见的差值（即公差）
    most_common_diff = Counter(diffs).most_common(1)[0][0]

    # 修正错误项
    for i in range(1, len(sequence)):
        expected_value = sequence[0] + i * most_common_diff  # 按公差计算的期望值
        if abs(sequence[i] - expected_value) > 1e-2:  # 判断是否偏差过大
            sequence[i] = expected_value  # 修正错误值

    return sequence


def get_pix_value_x(pix, x_values, x_positions):
    pix_x_value = 0
    x_list = [p[0] for p in x_positions]  # 这是 x 轴数字的横向像素位置坐标
    pix_x = pix[0]
    for i in (range(len(x_list)-1)):
        if x_list[i] <= pix_x <= x_list[i+1]:
            pix_x_value = x_values[i] + (x_values[i+1] - x_values[i]) * (pix_x - x_list[i]) / (x_list[i+1] - x_list[i])
            break

    return pix_x_value


def get_pix_value_y(pix, max_y_position, min_y_position):
    # 只考虑 y 坐标的相对位置
    pix_y = pix[1]
    return 1 - (pix_y - min_y_position) / (max_y_position - min_y_position)  # 像素扫描是从上往下的，但是图像坐标从下往上


def check_curve(color_dict, image_path):
    # 加载图像
    image = Image.open(image_path)
    width, height = image.width, image.height

    # 创建一个新的图像
    image = Image.new('RGB', (width, height), (255, 255, 255))  # 白色背景

    # 遍历字典并设置像素点
    for color, positions in color_dict.items():
        for position in positions:
            y, x = position
            image.putpixel((x, y), color)

    image.show()


def strengthen_image(image_path, strengthen_image_path, filled_image_path, darkness_limit=100):
    # 1. 读取图像
    image = cv2.imread(image_path)

    # 2. 创建一个输出图像，初始化为全白
    output_image = np.full_like(image, 255)  # 创建一个全白图像

    # 3. 找到需要替换的像素位置
    grayscale = np.mean(image, axis=2)  # 计算每个像素的灰度值
    mask_black = (grayscale <= darkness_limit)  # 找到灰度值符合条件的像素

    # 4. 找到彩色像素位置
    mask_color = (image[:, :, 0] != image[:, :, 1]) | (image[:, :, 1] != image[:, :, 2])

    # 5. 设置黑色像素和彩色像素的替换规则
    output_image[mask_black] = [0, 0, 0]  # 黑色像素保留为黑色
    output_image[mask_color] = [255, 255, 255]  # 彩色像素替换为白色

    # 6. 增强黑色像素的对比度（对所有黑色像素操作）
    alpha = 1.5  # 对比度增益因子
    beta = -50  # 亮度增益因子
    black_pixels = output_image[:, :, 0] == 0  # 找到黑色像素的掩码
    output_image[black_pixels] = np.clip(alpha * output_image[black_pixels] + beta, 0, 255)

    # 7. 保存增强后的图像
    cv2.imwrite(strengthen_image_path, output_image)

    # 8. 将图像二值化
    gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

    # 9. 使用形态学闭运算填充空洞
    kernel = np.ones((3, 3), np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # 10. 将图像反转为原来的黑白
    closed_image = cv2.bitwise_not(closed_image)

    # 11. 保存填充后的图像
    cv2.imwrite(filled_image_path, closed_image)


def extract_legend(curve_color_positions, filled_image_path):
    # 创建 OCR 对象
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        show_log=False,
        rec_model_dir='E:\\XAS\\pythonProject\\ch_PP-OCRv4_rec_infer'
    )

    # 调用 OCR，提取文字和坐标
    result = ocr.ocr(filled_image_path)
    ocr_bbox = [
        (res[0], res[1][0]) for res in result[0]
        if not is_numeric_except_hyphen_and_dot(res[1][0])
    ]
    bbox_coords = np.array([b[0] for b in ocr_bbox], dtype=float)  # 加速计算

    # 将曲线颜色与文字的边界框匹配
    curve_bbox_mapping = map_curve_to_bbox(curve_color_positions, bbox_coords)

    # 将 OCR 结果附加到 curve_bbox_mapping
    curve_bbox_mapping = [
        (ocr_bbox[entry[0]][1], entry[1], entry[2]) for entry in curve_bbox_mapping
    ]
    print('curve_bbox_mapping:', curve_bbox_mapping)

    # 处理最近匹配逻辑
    curve_nearest = {}  # 最终的颜色到文本的映射
    assigned_texts = set()  # 已经分配的文本
    assigned_curves = set()  # 已经分配的曲线

    # 先对所有 curve_bbox_mapping 按距离排序
    sorted_mapping = sorted(curve_bbox_mapping, key=lambda x: x[2])

    for text, curve, distance in sorted_mapping:
        # 如果该文本已经分配，跳过
        if text in assigned_texts:
            continue

        # 如果曲线未被分配，则直接分配
        if curve not in assigned_curves:
            curve_nearest[curve] = text
            assigned_texts.add(text)
            assigned_curves.add(curve)
        else:
            # 尝试为该文本寻找次近的曲线
            for alt_curve in sorted(curve_color_positions.keys()):
                if alt_curve not in assigned_curves and text not in assigned_texts:
                    curve_nearest[alt_curve] = text
                    assigned_texts.add(text)
                    assigned_curves.add(alt_curve)
                    break

    print('unchecked curve_nearest:', curve_nearest)
    # 检查规则：确保键值对符合边界框与曲线编号的对应关系
    swap_candidates = {}  # 记录需要交换的曲线对

    for curve, text in curve_nearest.items():
        # 获取对应文字的 bbox
        bbox_index = [i for i, ocr in enumerate(ocr_bbox) if ocr[1] == text][0]
        bbox_points = np.vstack((bbox_coords[bbox_index], np.mean(bbox_coords[bbox_index], axis=0)))

        # 相邻曲线范围
        adjacent_curves = [
            curve - 1 if curve > 0 else curve,
            curve,
            curve + 1 if curve + 1 in curve_color_positions else curve
        ]

        votes = {adj_curve: 0 for adj_curve in adjacent_curves}

        for point in bbox_points:
            distances = {
                adj_curve: np.min(np.linalg.norm(np.array(curve_color_positions[adj_curve]) - point, axis=1))
                for adj_curve in adjacent_curves
            }
            closest_curve = min(distances, key=distances.get)
            votes[closest_curve] += 1

        # 检查是否超过两个点更接近其他曲线
        for adj_curve, vote_count in votes.items():
            if adj_curve != curve and vote_count > 2:
                if curve in swap_candidates:
                    if swap_candidates[curve] != adj_curve:
                        raise ValueError(f"Conflicting swap rules between curve {curve} and curve {swap_candidates[curve]}")
                swap_candidates[curve] = adj_curve

    # 记录已处理的交换对，避免重复
    processed_swaps = set()

    # 交换键值对，确保双向交换只执行一次
    for curve, swap_curve in list(swap_candidates.items()):
        if swap_curve in swap_candidates and swap_candidates[swap_curve] == curve:
            if (curve, swap_curve) not in processed_swaps and (swap_curve, curve) not in processed_swaps:
                # 执行一次交换
                curve_nearest[curve], curve_nearest[swap_curve] = curve_nearest[swap_curve], curve_nearest[curve]
                # 记录处理过的交换对
                processed_swaps.add((curve, swap_curve))
                print(f"swap: curve {curve} -> curve {swap_curve}")
        else:
            # 单向交换需求，跳过
            print(f"Skipping unidirectional swap: curve {curve} -> curve {swap_curve}")

    return curve_nearest


def store_data(data, image_path, output_folder):
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    # 创建一个空的 DataFrame
    df = pd.DataFrame()

    # 遍历字典，将数据添加到 DataFrame 中
    for key, (x_data, y_data) in data.items():
        # 添加 x_data
        df[key + '_xdata'] = pd.Series(x_data)
        # 添加 y_data
        df[key + '_ydata'] = pd.Series(y_data)

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(f'{output_folder}/{file_name}_data.csv', index=False, encoding='gbk')


def get_intersection_union(list_of_lists):
    """
    获取列表中两两集合的交集，并返回所有交集的并集。
    param list_of_lists: 列表，其中每个元素是一个子列表或可转为集合的数据
    return: 所有交集的并集
    """
    # 将列表中的每个子列表预先转换为集合
    sets = [set(lst) for lst in list_of_lists]
    union_of_intersections = set()  # 初始化并集为空集合

    for set1, set2 in combinations(sets, 2):
        # 计算交集并累加到并集
        intersection = set1 & set2
        if intersection:  # 如果交集非空，才更新
            union_of_intersections.update(intersection)

    return union_of_intersections


def extract_data(image_path, custom_config, output_data_folder, output_curve_dir, strengthen_image_path, filled_image_path):
    # Part 1
    color_position_dict = extract_color_position_dict(image_path)
    final_points_dict, curve_colors = extract_curve(image_path, output_curve_dir, visualize=False)

    # Part 2
    color_intersection = get_intersection_union([colors for colors in curve_colors.values()])
    # 获取和曲线颜色相同的像素点的位置列表，方便提取图例
    # 提前处理数据
    color_to_curve = {}
    for curve_id, colors in curve_colors.items():
        for color in colors:
            color_to_curve[color] = curve_id
    # 提取有效颜色
    valid_colors = set(color_position_dict.keys()) - color_intersection
    # 初始化结果字典
    curve_color_positions = defaultdict(list)
    # 主循环
    for color in valid_colors:
        if color in color_to_curve:
            curve_id = color_to_curve[color]
            curve_color_positions[curve_id] += color_position_dict[color]
    # 转换为普通字典并去重位置
    curve_color_positions = {curve_id: list(set(positions)) for curve_id, positions in curve_color_positions.items()}

    # Part 3
    strengthen_image(image_path, strengthen_image_path, filled_image_path)
    x_values, x_positions = extract_axis_values(filled_image_path, custom_config)
    x_values = fix_arithmetic_sequence(x_values)
    print('x_values, x_positions:', x_values, x_positions)

    # Part 4
    max_y_position, min_y_position = get_max_and_min_y(final_points_dict)
    print(max_y_position, min_y_position)
    curve_data_dict = {}
    for curve_id, positions in final_points_dict.items():
        curve_data_dict[curve_id] = []
        for p in positions:
            pix_x_value = get_pix_value_x(p, x_values, x_positions)
            pix_y_value = get_pix_value_y(p, max_y_position, min_y_position)
            if pix_x_value != 0:
                curve_data_dict[curve_id].append((pix_x_value, pix_y_value))

    # Part 5 *
    curve_nearest = extract_legend(curve_color_positions, filled_image_path)
    print('curve_nearest:', curve_nearest)


    # Part 6
    data = {}
    for curve_id in curve_data_dict.keys():
        x = [v[0] for v in curve_data_dict[curve_id]]
        y = [v[1] for v in curve_data_dict[curve_id]]
        # 将 x 和 y 组合成元组对，然后按 x 排序
        sorted_pairs = sorted(zip(x, y), key=lambda pair: pair[0])
        # 解压排序后的结果
        x_sorted, y_sorted = zip(*sorted_pairs)
        # 转换为列表（可选）
        x_sorted = list(x_sorted)
        y_sorted = list(y_sorted)
        print(f"{curve_id} - x length: {len(x_sorted)}, y length: {len(y_sorted)}")  # 输出长度
        legend = curve_nearest[curve_id].replace(';', '3').replace('；', '3')  # 特殊处理
        data[legend] = [x_sorted, y_sorted]
    store_data(data, image_path, output_data_folder)


if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract.exe'
    image_path = './example_fig5_part_1_part_2.png'
    custom_config = '--oem 3 --psm 11'
    output_data_folder = 'output_data'
    output_curve_dir = 'output_images'
    strengthen_image_path = 'processed_image.png'
    filled_image_path = 'filled_image.png'

    clear_and_recreate_directory(output_data_folder)
    clear_and_recreate_directory(output_curve_dir)

    extract_data(image_path, custom_config, output_data_folder, output_curve_dir, strengthen_image_path, filled_image_path)

