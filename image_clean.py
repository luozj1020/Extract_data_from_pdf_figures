import numpy as np
from PIL import Image
import os
import pytesseract

from string_operation import compare_strings_ratio
from extract_image import ocr_image, crop_header, clear_and_recreate_directory


def gaussian(x, x0, width, a=1):
    # 调整 sigma 使得在 x0 + width 处函数值接近 0
    sigma = width / np.sqrt(2 * np.log(100))  # 让函数在 x0 + width 处接近 0
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def trim_whitespace(image):
    # 转换为RGB模式以确保我们处理的是三通道图像
    image = image.convert("RGB")

    # 将图像转换为numpy数组
    image_array = np.array(image)

    # 创建一个布尔数组，表示非白色区域 (255, 255, 255)
    non_white_mask = np.all(image_array != [255, 255, 255], axis=-1)

    # 找到非白色区域的边界
    non_empty_columns = np.where(np.any(non_white_mask, axis=0))[0]
    non_empty_rows = np.where(np.any(non_white_mask, axis=1))[0]

    if non_empty_columns.size and non_empty_rows.size:
        # 获取边界框
        left = non_empty_columns[0]
        correct = non_empty_columns[-1] + 1
        top = non_empty_rows[0]
        bottom = non_empty_rows[-1] + 1

        # 裁剪图像
        # print((left, top, correct, bottom))
        cropped_image = image.crop((left, top, correct, bottom))
        return cropped_image, left
    else:
        print("图片全是空白")
        return image, 0


def load_images_from_folder(folder_path):
    images = []
    fig_names = []
    for filename in os.listdir(folder_path):
        fig_names.append(filename)
        # 构造完整路径
        file_path = os.path.join(folder_path, filename)
        # 判断是否为图片文件（可以根据扩展名过滤）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                img = Image.open(file_path)
                images.append(img)
            except Exception as e:
                print(f"无法打开图片 {filename}: {e}")
    return images, fig_names


def get_x_position(images, fig_names, custom_config):
    x_positions = {}
    correct_images = {}
    for image in images:
        fig_name = fig_names[images.index(image)].replace('.png', '')
        print('fig_name:', fig_name)
        tmp = []
        ocr_data = ocr_image(image, custom_config)
        # print(ocr_data['text'])
        x_symbol = ['E', 'eV', 'Energy']
        for i, text in enumerate(ocr_data['text']):
            if text != '':
                for s in x_symbol:
                    if len(s)==1 and s==text or\
                        len(s)>1 and len(text)>1 and abs(len(s) - len(text)) < 3 and compare_strings_ratio(text.lower(), s.lower()) > 0.9:
                        print('text, s:', (text, s))
                        tmp.append((ocr_data['left'][i], ocr_data['width'][i], ocr_data['top'][i], ocr_data['height'][i]))
        if tmp != []:
            correct_images[fig_name] = image
            x_positions[fig_name] = tmp
    print('x_positions:', x_positions)
    return correct_images, x_positions


def merge_tuples(tuples_list, threshold, pos=0):
    merged = True

    while merged:
        merged = False
        new_tuples = []
        skip = False

        for i in range(len(tuples_list)):
            if skip:
                skip = False
                continue

            for j in range(i + 1, len(tuples_list)):
                if abs(tuples_list[i][pos] - tuples_list[j][pos]) < threshold:
                    new_tuple = tuple(int((tuples_list[i][k] + tuples_list[j][k]) / 2) for k in range(len(tuples_list[i])))
                    new_tuples.append(new_tuple)
                    skip = True
                    merged = True
                    break

            if not skip:
                new_tuples.append(tuples_list[i])

        tuples_list = new_tuples

    return tuples_list


def find_similar_numbers_with_tuples(numbers_with_tuples, img_width, img_height):
    # 按数字进行排序，保留关联元组
    sorted_numbers_with_tuples = sorted(numbers_with_tuples, key=lambda x: x[0])
    result = []
    visited = set()  # 用于跟踪已经处理的数字
    threshold = 0.15 * img_height  # 定义相似的阈值

    for i in range(len(sorted_numbers_with_tuples)):
        number, associated_tuple = sorted_numbers_with_tuples[i]

        if number in visited:
            continue  # 跳过已经处理过的数字

        # 初始化一个相似数字组
        similar_group = [(number, associated_tuple)]
        visited.add(number)

        # 查找相似的数字
        for j in range(i + 1, len(sorted_numbers_with_tuples)):
            next_number, next_tuple = sorted_numbers_with_tuples[j]

            if abs(next_number - number) <= threshold:
                similar_group.append((next_number, next_tuple))
                visited.add(next_number)
            else:
                break  # 因为数组已排序，后面的数字必定更大，直接退出

        # 找到相似数字组中最大的数字及其关联的元组
        max_number, max_tuple = max(similar_group, key=lambda x: x[0])

        # 收集所有相关的元组
        associated_tuples = [tup[0:2] for _, tup in similar_group]

        # 检查这个最大值是否有相似数字（至少要有两个数字）
        if len(similar_group) > 1:
            result.append((int(max_number+0.5*max_tuple[3]), merge_tuples(associated_tuples, 0.2*img_width)))

    print('l_t:', result)
    return [r[0] for r in result], result


def get_x_bottom(x_positions, img_width, img_height):
    x_bottoms, l_t = find_similar_numbers_with_tuples([(p[2]+p[3], p) for p in x_positions], img_width, img_height)
    return x_bottoms, l_t


# 竖向分割图片
def split_image_multiple_y(img, split_y_coordinates, save_path, fig_name):
    # 清除残留的（表格）横线
    img = crop_header(img, 0.85)  # 可能上方有表格存在，用截页眉的方法去掉
    if isinstance(img, np.ndarray):  # 检查 img 是否为 NumPy 数组
        img = Image.fromarray(img)  # 将 NumPy 数组转换为 Pillow 图像
    # 获取图片的宽度和高度
    width, height = img.size

    # 处理纵坐标列表，确保其有序且在有效范围内
    split_y_coordinates = sorted([min(y, height) for y in split_y_coordinates])

    if not split_y_coordinates:
        pass
        # raise ValueError("没有有效的纵坐标可用于分割。")

    # 初始化分割部分的列表
    split_images = []
    cropped_parts_left = []
    last_y = 0  # 上一个分割纵坐标

    for split_y in split_y_coordinates:
        # 分割图片
        part, left = trim_whitespace(img.crop((0, last_y, width, split_y)))
        cropped_parts_left.append(left)
        split_images.append(part)  # 这里对图片进行去除边缘空白处的操作，为了防止下面判断图片中共有几张图的时候发生错误
        last_y = split_y  # 更新上一个纵坐标

    # 保存分割后的图片
    split_img_names = []
    for index, part in enumerate(split_images):
        split_img_names.append(str(f"{fig_name}_part_{index + 1}"))
        part.save(f"./{save_path}/{fig_name}_part_{index + 1}.png")

    return split_images, cropped_parts_left, split_img_names


def check_img_num(x_list, gauss_mean_list, img_num, img_width):
    # position 为元组，第一个值表示左侧位置，第二个位置表示长度
    check_count = 0
    for x in x_list:
        for x0 in gauss_mean_list:
            # print((x[0]+x[1]/2, x0, img_width), gaussian(x[0]+x[1]/2, x0, img_width))
            if gaussian(x[0]+x[1]/2, x0, img_width) >= 0.85:
                check_count += 1
    if check_count == len(x_list):
        return True
    else:
        return False



# 确定有几张图片
def count_figs_x(x_list, img_width):
    img_num = len(x_list)
    check = False
    while not check:
        gauss_mean_list = [i*img_width/(img_num+1) for i in range(1, img_num+1)]
        # print((x_list, gauss_mean_list, img_num, img_width))
        check = check_img_num(x_list, gauss_mean_list, img_num, img_width)
        # print(check)

        if check:
            break
        else:
            img_num += 1

    return img_num


def save_image_parts_x(images, folder_path, output_prefix, drop_out=True):
    """
    将分割后的图像列表保存到指定文件夹中。
    """
    for i, img in enumerate(images):
        if drop_out:
            if img.height > img.width:
                print(f"图像丢弃")
            else:
                output_path = os.path.join(folder_path, f"{output_prefix}_part_{i + 1}.png")
                img.save(output_path)
                print(f"图像保存到: {output_path}")
        else:
            output_path = os.path.join(folder_path, f"{output_prefix}_part_{i + 1}.png")
            img.save(output_path)
            print(f"图像保存到: {output_path}")


def check_split_x(pos_list, previous_split, split_point):
    check = False
    for p in pos_list:
        if previous_split <= p <= split_point:
            check = True
            return check
    return check


def split_and_save_image_x(image, num_images, pos_list, folder_path, output_prefix):
    """
    根据图像数量进行分割并保存，如果图像数量为1，则直接保存原图。
    """
    image_np = np.array(image)

    # 如果图像数量是1，直接保存
    if num_images == 1:
        save_image_parts_x([image], folder_path, output_prefix)
        return

    # 转换为灰度图以简化空白区域检测
    grayscale_image = np.mean(image_np, axis=2)

    # 获取图像的宽度和高度
    height, width = grayscale_image.shape

    # 通过检查像素强度找到竖直方向的空白列（连续的空白区域）
    blank_threshold = 250  # 根据需要调整此值以检测空白区域
    blank_columns = np.where(np.mean(grayscale_image, axis=0) > blank_threshold)[0]

    if len(blank_columns) > 0:
        # 假设你告诉我们有几张图像，我们需要找到 num_images - 1 个分割点
        split_points = []
        approximate_width = width // num_images  # 每张图像的近似宽度

        # 遍历所有空白列，找到最接近每张图像边界的空白列
        for i in range(1, num_images):
            # 查找最接近于 (i * approximate_width) 位置的空白列
            closest_blank = min(blank_columns, key=lambda x: abs(x - i * approximate_width))
            split_points.append(closest_blank)

        # 将图像分割为 num_images 部分
        previous_split = 0
        part_images = []
        for split_point in split_points + [width]:  # 加上图像的最后一个边界
            part_image = Image.fromarray(image_np[:, previous_split:split_point])
            if check_split_x(pos_list, previous_split, split_point):
                part_images.append(part_image)
            previous_split = split_point  # 更新上一次的分割点

        # 保存分割后的图像
        save_image_parts_x(part_images, folder_path, output_prefix)
    else:
        print("未找到显著的空白区域进行分割。")


def clean_img(folder_path, custom_config, save_path_y, save_path_x):

    clear_and_recreate_directory(save_path_y)
    clear_and_recreate_directory(save_path_x)

    images, fig_names = load_images_from_folder(folder_path)
    correct_images, x_positions = get_x_position(images, fig_names, custom_config)
    for fig_name, x_positions in x_positions.items():
        x_bottoms, l_t = get_x_bottom(x_positions, correct_images[fig_name].width, correct_images[fig_name].height)
        # 每个 split_image 对应 l_t 中一个元组元素
        split_images, cropped_parts_left, split_img_names = split_image_multiple_y(correct_images[fig_name], x_bottoms, save_path_y, fig_name)
        # 需要确定的是 split_image 对应的元组 t 中图像的个数，也就是说，完成纵向的切割后，要进行横向切割前，要判断列表 t[1] 的元素个数是否就是横向切割出来的图的个数
        for i in range(len(l_t)):
            split_image = split_images[i]
            part_left = cropped_parts_left[i]
            x_list = l_t[i][1]
            # print('x_list:', x_list)
            # 更新 x 标记的位置，因为前面裁掉了空白的部分
            x_list = [tuple((int(t[0]-part_left), t[1])) for t in x_list]
            # print('new x_list:', x_list)
            split_img_name = split_img_names[i]
            img_num = count_figs_x(x_list, split_image.width)
            split_and_save_image_x(split_images[i], img_num, [p[0]+p[1]/2 for p in x_list], save_path_x, split_img_name)



if __name__ == '__main__':
    # 配置 tesseract 可执行文件路径（如果需要）
    pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract.exe'
    custom_config = r'--oem 3 --psm 11' # 使用原始Tesseract OCR引擎和神经网络LSTM引擎的结合；只识别文本
    folder_path = 'fig_output'
    save_path_y = 'cleaned_figs_y'
    save_path_x = 'cleaned_figs_x'
    clean_img(folder_path, custom_config, save_path_y, save_path_x)



