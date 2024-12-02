import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import json
import cv2
import numpy as np
import os
import shutil

from string_operation import *


# 读取PDF并将每一页转换为图片
def pdf_to_images(pdf_path, dpi=500):
    return convert_from_path(pdf_path, dpi=dpi)


def crop_header(image, threshold_power):
    # 将PIL图像转换为OpenCV格式
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 转换为灰度图
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊，减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 获取图像宽度
    image_width = open_cv_image.shape[1]
    header_y = 0
    header_h = 0

    # 筛选出可能的页眉区域
    for contour in contours:
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 假设页眉的高度较小，宽度接近半个页面宽度或整个页面宽度
        if h < 100 and (w > image_width * threshold_power and w <= image_width):
            header_y = y
            header_h = h

    # 如果找到了页眉，截取下面部分
    if header_h > 0:
        cropped_image = open_cv_image[header_y + header_h:open_cv_image.shape[0], :]
        # 去除上方的绿线
        cropped_image = cropped_image[10:]  # 去掉前几行（绿线所在行）
        return cropped_image
    else:
        print('没找到页眉')
        return image


# 对图片进行OCR识别并返回每段文字的位置和内容
def ocr_image(image, custom_config):
    # 使用 pytesseract 识别文字
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)
    return data


def get_fig_paragraph(pdf_path):
    para_dict = {}
    page_dict = {}

    # 打开PDF文件
    doc = fitz.open(pdf_path)
    # 遍历每一页
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()  # 提取文本
        page_dict[page_num] = process_text(text)
        # 识别开头可能是图片说明的段落
        text_blocks = page.get_text("blocks")  # 获取文本块
        possible_para = []

        for block in text_blocks:
            text = block[4].strip()  # 提取文本并去除前后的空格
            # 使用正则表达式提取以 "Fig." 或 "Figure" 开头，并紧跟数字的段落
            match = re.match(r'^(fig(?:ure)?\.?\s*(\d+))', text, re.IGNORECASE)
            if match:
                fig_number = match.group(2)  # 提取数字部分
                figure_description = process_text(text)
                # 如果段落开头是Fig. 的形式
                possible_para.append((fig_number, figure_description))

        para_dict[page_num] = possible_para
    return page_dict, para_dict


def get_back_word(page_text, possible_para):
    # 先找到上沿的位置，之后在ocr中匹配
    back = 1
    k = 3
    back_place = find_sublist_index(page_text.split(' '), possible_para.split(' '))

    possible_back_word = ''
    while back_place - back > 0:
        if is_alphanumeric_or_punctuation(page_text.split(' ')[back_place - back][-1]):
            possible_back_word = page_text.split(' ')[back_place - back]
            if back_place - back > k:
                possible_back_word = (
                        "".join(
                            [page_text.split(' ')[back_place - back - j - 1] for j in range(k)])
                        + possible_back_word)
                break
        back += 1
    print(possible_back_word)
    return possible_back_word


# 查找包含fig的段落，返回其上沿和下沿
def find_figure_boundaries(back_words, p, ocr_data):
    fig_pattern = re.compile(r'^(fig(?:ure)?\.?\s*\d+)', re.IGNORECASE)
    boundaries = []

    for i, text in enumerate(ocr_data['text']):
        check = 5
        if text != '':
            if i + check < len(ocr_data['text']):
                possible_start = remove_non_alphanumeric("".join([ocr_data['text'][i + j] for j in range(check)]))
            else:
                possible_start = remove_non_alphanumeric("".join([ocr_data['text'][j] for j in range(i,len(ocr_data['text']))]))

            # 考虑到ocr和pdf文字提取并不完全准确，不再采用严格匹配
            if compare_strings_ratio(possible_start, remove_non_alphanumeric(p)) > 0.8 and fig_pattern.search(possible_start):
                print((i, len(ocr_data['text']), remove_non_alphanumeric(p), possible_start, compare_strings_ratio(possible_start, remove_non_alphanumeric(p))))
                # 获取fig段落的上沿坐标
                fig_top = ocr_data['top'][i]
                # 获取fig段落的左沿坐标
                fig_left = ocr_data['left'][i]
                # 获取fig段落的右沿坐标
                words_list = p.split(' ')
                record_position = []
                for j in range(len(words_list)):
                    if not words_list[j].isdigit() and len(words_list[j]) > 1:
                        for k in range(i, i+len(words_list)):
                            if abs(len(words_list[j]) - len(ocr_data['text'][k])) <= 3 and compare_strings_ratio(ocr_data['text'][k], words_list[j]) > 0.9:
                                # print(ocr_data['text'][k], words_list[j], ocr_data['left'][k], ocr_data['width'][k])
                                record_position.append(ocr_data['left'][k] + ocr_data['width'][k])
                fig_right = max(record_position)
                # 获取fig段落上一段文字的下沿坐标（如果有）
                if back_words == '':
                    upper_text_bottom = 0
                else:
                    back = 1
                    upper_text_bottom = 0
                    while i - back > 0:
                        if ocr_data['text'][i-back] and is_alphanumeric_or_punctuation(ocr_data['text'][i-back][-1]) and len(ocr_data['text'][i-back]) > 3:
                            check_back_word = ocr_data['text'][i-back].replace(' ', '')

                            if compare_strings_ratio(check_back_word[::-1], remove_non_alphanumeric(back_words[::-1])) > 0.8:
                                upper_text_bottom = ocr_data['top'][i - back] + ocr_data['height'][i - back]
                                print((check_back_word, back_words, compare_strings_ratio(check_back_word[::-1], remove_non_alphanumeric(back_words[::-1]))))
                                if upper_text_bottom > fig_top:
                                    upper_text_bottom = 0
                                break
                        back += 1

                boundaries.append((fig_left, upper_text_bottom, fig_right, fig_top))
    return boundaries


# 根据找到的上下沿裁剪图片
def crop_image(image, boundaries):
    k = 10
    cropped_images = []

    # Ensure the image is a NumPy array for cv2 processing
    if not isinstance(image, np.ndarray):
        # Convert the PIL Image (or other format) to a NumPy array if necessary
        image = np.array(image)

    for (left, upper, right, lower) in boundaries:
        if abs(upper - lower) > 50:
            # Convert the image to RGB format
            image_rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            cropped = image_rgb.crop((left + k, upper + k, right - k, lower - k))
            cropped_images.append(cropped)

    return cropped_images


# 保存裁剪后的图片
def save_images(pdf_name, fig_num, images, output_folder):
    for i, img in enumerate(images):
        enhancer = ImageEnhance.Contrast(img)
        enhanced_image = enhancer.enhance(factor=2.0)  # 增强对比度
        enhanced_image.save(f"{output_folder}/{pdf_name}_fig{fig_num}.png")


# 主函数
def extract_figures_from_pdf(pdf_path, json_path, output_folder, custom_config, keywords_list):
    # 读取之前生成图片描述的 JSON 文件
    with open(json_path, "r", encoding="utf-8") as json_file:
        figures_descriptions = json.load(json_file)
    images = pdf_to_images(pdf_path)
    page_dict, para_dict = get_fig_paragraph(pdf_path)
    print(para_dict)
    for page_num, image in enumerate(images):
        if para_dict[page_num] != []:
            # 截掉页眉
            cropped_header_image = crop_header(image, 0.4)
            ocr_data = ocr_image(cropped_header_image, custom_config)
            for p in para_dict[page_num]:
                possible_para = p[1]
                for d in figures_descriptions[p[0]]:
                    if is_keywords(d, keywords_list):
                        back_words = get_back_word(page_dict[page_num], possible_para)
                        # print(back_words)
                        figure_boundaries = find_figure_boundaries(back_words, possible_para, ocr_data)
                        print(figure_boundaries)
                        cropped_images = crop_image(cropped_header_image, figure_boundaries)
                        save_images(pdf_path.replace('.pdf', ''), p[0], cropped_images, output_folder)
                        break


def clear_and_recreate_directory(folder_path):
    # 删除整个文件夹
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    # 重新创建空文件夹
    os.makedirs(folder_path)
    print(f"已清空并重新创建目录 '{folder_path}'")


if __name__ == '__main__':
    keywords_list = ['XANES', 'XAS', 'EXAFS', 'XMCD', 'XMCD', 'XLD', 'RIXS', 'X-ray']
    # 配置 tesseract 可执行文件路径（如果需要）
    pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract.exe'
    custom_config = r'--oem 3'
    # 使用示例
    pdf_path = "example1.pdf"
    output_folder = "fig_output"
    json_path = 'figures_descriptions.json'

    clear_and_recreate_directory(output_folder)
    extract_figures_from_pdf(pdf_path, json_path, output_folder, custom_config, keywords_list)
