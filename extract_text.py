import fitz  # PyMuPDF
import re
import json
from deep_translator import GoogleTranslator
from collections import Counter


# 定义处理换行符和特殊字符的函数
def process_text(text):
    text = text.replace('ﬁ', 'fi')  # 替换特殊字符
    # 替换换行符
    text = re.sub(r'\n(?=\w|[.!?])', ' ', text)  # 换行后是字母、数字或句子结束符号时替换为空格
    text = re.sub(r'\n', '', text)  # 去掉其他换行符
    return text


# 定义函数去掉不完整或以小写字母开头的句子
def remove_incomplete_or_lowercase_sentences(text):
    # 如果文本以 "Fig." 或 "Figure" 开头，则返回原文本
    if re.match(r'^(fig\.|figure)', text, re.IGNORECASE):
        return text

    # 使用正则表达式分割句子，考虑句号、问号、感叹号和括号
    sentences = re.split(r'(?<=[.!?]) +|(?<=\)) +', text)
    filtered_sentences = [
        s for s in sentences if s and s[0].isupper()
    ]

    return ' '.join(filtered_sentences)


# 获取关于图片的描述并存储到json文件，并且对描述进行翻译，存储在中英文对照的json文件中
def get_fig_description(pdf_path):
    # 存储图像描述的字典
    figures = {}

    # 打开PDF文件
    doc = fitz.open(pdf_path)
    # 定义正则表达式用于匹配图像描述
    fig_pattern = re.compile(r'(?i)[^\w]*(fig|figure)[^\w]*([0-9]+)')
    # 遍历每一页
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_blocks = page.get_text("blocks")

        for i, block in enumerate(text_blocks):
            text = block[4].strip()  # 提取文本并去除前后的空格

            # 如果是这一页的最后一个block，检查条件并补充下一页的第一个block
            if i == len(text_blocks) - 1:  # 最后一个块
                if len(text) < 20 or not text.endswith(('.', '!', '?')):  # 检查长度和结束符
                    if page_num + 1 < len(doc):  # 检查是否有下一页
                        next_page = doc[page_num + 1]
                        next_text_blocks = next_page.get_text("blocks")
                        if next_text_blocks:  # 确保下一页有内容
                            text += " " + next_text_blocks[0][4].strip()  # 添加下一页的第一个块
                            text = process_text(text)  # 处理文本
                            text = remove_incomplete_or_lowercase_sentences(text)  # 去掉不完整或以小写字母开头的句子

            # 查找图像描述
            matches = fig_pattern.finditer(text)
            for match in matches:
                fig_label = match.group(1)  # "fig" 或 "figure"
                fig_num = match.group(2)  # 图像编号

                # 找到“fig”或“figure”的起始位置
                fig_index = match.start(1)

                # 检查上下文是否包含“supp”或“Supplementary”
                context = text.lower()[max(0, fig_index - 30):fig_index + 30]  # 获取上下文
                if "supp" in context or "supplementary" in context:
                    continue  # 如果上下文包含“supp”或“Supplementary”，则跳过

                if fig_num not in figures:
                    figures[fig_num] = []  # 使用列表存储文本块
                figures[fig_num].append(text)  # 添加描述文本块

    # 去重
    for fig_num in figures:
        figures[fig_num] = list(set(figures[fig_num]))  # 转换为集合再回列表去重

    # 保存结果到JSON文件
    with open("figures_descriptions.json", "w", encoding="utf-8") as json_file:
        json.dump(figures, json_file, ensure_ascii=False, indent=4)

    # 关闭PDF文件
    doc.close()

    # 读取之前生成的 JSON 文件
    with open("figures_descriptions.json", "r", encoding="utf-8") as json_file:
        figures = json.load(json_file)

    # 创建一个新的字典以存储中英文对照
    bilingual_figures = {}

    # 遍历每个图像编号及其描述
    for fig_num, descriptions in figures.items():
        bilingual_figures[fig_num] = []
        for description in descriptions:
            # 翻译英文描述
            translated = GoogleTranslator(source='en', target='zh-CN').translate(description)
            bilingual_figures[fig_num].append({
                "english": description,
                "chinese": translated
            })

    # 保存中英文对照的结果到新的 JSON 文件
    with open("bilingual_figures_descriptions.json", "w", encoding="utf-8") as json_file:
        json.dump(bilingual_figures, json_file, ensure_ascii=False, indent=4)

    print("翻译完成，中英文对照已保存。")


def most_common_element(lst):
    # 使用 Counter 统计元素出现的次数
    count = Counter(lst)
    # 使用 most_common 获取出现次数最多的元素，返回一个 (元素, 次数) 元组
    return count.most_common(1)[0][0]


if __name__ == '__main__':
    pdf_path = 'example.pdf'
    get_fig_description(pdf_path)