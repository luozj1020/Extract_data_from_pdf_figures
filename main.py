from extract_text import *
from extract_image import *
from image_clean import *
from extract_data import *

import time

if __name__ == '__main__':
    keywords_list = ['XANES', 'XAS', 'EXAFS', 'XMCD', 'XMCD', 'XLD', 'RIXS', 'X-ray']
    now = time.time()
    # 使用示例
    # 配置 tesseract 可执行文件路径（如果需要）
    pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract.exe'
    custom_config = r'--oem 3'  # 使用原始Tesseract OCR引擎和神经网络LSTM引擎的结合
    # custom_config = None
    pdf_path = "example.pdf"
    get_fig_description(pdf_path)  # 需要访问谷歌翻译

    output_folder = "fig_output"
    json_path = 'figures_descriptions.json'

    clear_and_recreate_directory(output_folder)
    extract_figures_from_pdf(pdf_path, json_path, output_folder, custom_config, keywords_list)

    custom_config = r'--oem 3 --psm 11'  # 使用原始Tesseract OCR引擎和神经网络LSTM引擎的结合；只识别文本
    folder_path = r'fig_output'
    save_path_y = r'cleaned_figs_y'
    save_path_x = r'cleaned_figs_x'
    clean_img(folder_path, custom_config, save_path_y, save_path_x)

    output_folder = 'output_data'
    custom_config = '--oem 3 --psm 11'
    output_data_folder = 'output_data'
    output_curve_dir = 'output_images'
    strengthen_image_path = 'processed_image.png'
    filled_image_path = 'filled_image.png'
    clear_and_recreate_directory(output_data_folder)
    clear_and_recreate_directory(output_curve_dir)

    # 指定文件夹路径
    folder_path = save_path_x  # 使用原始字符串（r'...'）来处理反斜杠
    # 列出文件夹下的所有文件，并获取完整路径
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                os.path.isfile(os.path.join(folder_path, f))]
    # 将文件路径作为输入传递给其他函数
    for file_path in tqdm(file_paths):
        extract_data(file_path, custom_config, output_data_folder, output_curve_dir, strengthen_image_path,
                    filled_image_path)

    print('cost time:', time.time() - now)