import re


def compare_strings_with_flexible_skips(str1, str2, max_skips=3):
    i, j = 0, 0  # 两个指针分别指向 str1 和 str2
    same_char_count = 0
    skips = 0  # 跳过字符的计数器

    # 遍历两个字符串
    while i < len(str1) and j < len(str2):
        if str1[i] == str2[j]:
            # 字符匹配成功，继续比较下一个字符
            same_char_count += 1
            i += 1
            j += 1
        else:
            # 尝试跳过 str2 中的字符（str1 保持不变）
            if skips < max_skips:
                j += 1
                skips += 1
            else:
                # 如果已经跳过 max_skips 次仍不匹配，跳过 str1 中的字符继续
                i += 1
                skips = 0  # 重置跳过计数器

    # 计算相似度，使用较短字符串的长度作为基准
    min_length = min(len(str1), len(str2))
    similarity_ratio = same_char_count / min_length if min_length > 0 else 0
    return similarity_ratio


def compare_strings_ratio(str1, str2, max_skips=3):
    return max(compare_strings_with_flexible_skips(str1, str2, max_skips=3), compare_strings_with_flexible_skips(str2, str1, max_skips=3))


def remove_non_alphanumeric(s):
    # 使用正则表达式替换所有非字母数字的字符
    return str(re.sub(r'[^a-zA-Z0-9]', '', s))


def is_alphanumeric_or_punctuation(char):
    # 正则表达式匹配字母、数字或句子结尾的标点
    return bool(re.match(r'^[a-zA-Z0-9.!?]$', char))


def find_sublist_index(main_list, sublist):
    sublist_length = len(sublist)
    main_length = len(main_list)

    # 遍历主列表，检查子列表是否匹配
    for i in range(main_length - sublist_length + 1):  # 确保有足够的空间
        # 使用切片检查匹配
        if main_list[i:i + sublist_length] == sublist:
            return i  # 返回匹配的起始位置

    return -1  # 如果未找到，返回 -1


def similarity_ratio(str1, str2):
    # 计算较短字符串的长度
    shorter_length = min(len(str1), len(str2))
    # 将两个字符串转换为集合，找到相同字符
    set1 = set(str1[0:shorter_length])
    set2 = set(str2[0:shorter_length])
    # 找到相同字符
    common_chars = set1.intersection(set2)
    # 计算相同字符的数量
    common_count = len(common_chars)
    # 计算比值，避免除以零
    if shorter_length == 0:
        return 0.0  # 如果任一字符串为空，返回比值为0

    ratio = float(common_count) / float(shorter_length)
    return ratio


# 定义处理换行符和特殊字符的函数
def process_text(text):
    text = text.replace('ﬁ', 'fi')  # 替换特殊字符
    # 替换换行符
    text = re.sub(r'\n(?=\w|[.!?])', ' ', text)  # 换行后是字母、数字或句子结束符号时替换为空格
    text = re.sub(r'\n', '', text)  # 去掉其他换行符
    return text


def is_keywords(text, keywords_list):
    # 判断图片描述是否和 XAS 相关
    check = False
    for w in keywords_list:
        if w.lower() in (text.lower()).split(' '):
            check = True
            break

    return check


def is_numeric_except_hyphen_and_dot(s):
    # 使用正则表达式检查字符串是否只包含数字、小数点和可选的负号
    return bool(re.fullmatch(r'.?\d+(\.\d+)?', s.replace('-', '')))


if __name__ == '__main__':
    # 测试示例
    string1 = "hello world"
    string2 = "hello world"

    result = compare_strings_ratio(string1, string2, max_skips=3)
    print(f"相似度: {result:.2f}")

    print(compare_strings_ratio(remove_non_alphanumeric('Fig. 4 Results of the similarity ranking returned by the ELSIE matching algorithm on aAl K-edge XANES of α-Al2O3entry; bNa K-edge XANES'), 'Fig4Resultsofthe'))
