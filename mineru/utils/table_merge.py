# Copyright (c) Opendatalab. All rights reserved.
import re
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

from loguru import logger
from bs4 import BeautifulSoup

from mineru.backend.vlm.vlm_middle_json_mkcontent import merge_para_with_text
from mineru.utils.char_utils import full_to_half
from mineru.utils.enum_class import BlockType, SplitFlag


CONTINUATION_MARKERS = [
    "(续)", "(续表)", "(续上表)", "(接上表)", "(承上)",
    "（续）", "（续表）", "（续上表）", "（接上表）", "（承上）",
    "续表", "续", "接上表", "承上",

    "(continued)", "(cont.)", "(cont'd)", "(continuation)",
    "continued", "cont.", "cont'd",
    "(Continued)", "(Cont.)", "(Cont'd)", "(Continuation)",
    "Continued", "Cont.", "Cont'd",
    
    "- continued", "- cont.", "...continued",
    "– continued", "— continued",
]

CONTINUATION_END_MARKERS = [
    "(续)",
    "(续表)",
    "(续上表)",
    "(continued)",
    "(cont.)",
    "(cont’d)",
    "(…continued)",
    "续表",
]

CONTINUATION_INLINE_MARKERS = [
    "(continued)",
]


def calculate_table_total_columns(soup):
    """计算表格的总列数，通过分析整个表格结构来处理rowspan和colspan

    Args:
        soup: BeautifulSoup解析的表格

    Returns:
        int: 表格的总列数
    """
    rows = soup.find_all("tr")
    if not rows:
        return 0

    # 创建一个矩阵来跟踪每个位置的占用情况
    max_cols = 0
    occupied = {}  # {row_idx: {col_idx: True}}

    for row_idx, row in enumerate(rows):
        col_idx = 0
        cells = row.find_all(["td", "th"])

        if row_idx not in occupied:
            occupied[row_idx] = {}

        for cell in cells:
            # 找到下一个未被占用的列位置
            while col_idx in occupied[row_idx]:
                col_idx += 1

            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            # 标记被这个单元格占用的所有位置
            for r in range(row_idx, row_idx + rowspan):
                if r not in occupied:
                    occupied[r] = {}
                for c in range(col_idx, col_idx + colspan):
                    occupied[r][c] = True

            col_idx += colspan
            max_cols = max(max_cols, col_idx)

    return max_cols


def build_table_occupied_matrix(soup):
    """构建表格的占用矩阵，返回每行的有效列数

    Args:
        soup: BeautifulSoup解析的表格

    Returns:
        dict: {row_idx: effective_columns} 每行的有效列数（考虑rowspan占用）
    """
    rows = soup.find_all("tr")
    if not rows:
        return {}

    occupied = {}  # {row_idx: {col_idx: True}}
    row_effective_cols = {}  # {row_idx: effective_columns}

    for row_idx, row in enumerate(rows):
        col_idx = 0
        cells = row.find_all(["td", "th"])

        if row_idx not in occupied:
            occupied[row_idx] = {}

        for cell in cells:
            # 找到下一个未被占用的列位置
            while col_idx in occupied[row_idx]:
                col_idx += 1

            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            # 标记被这个单元格占用的所有位置
            for r in range(row_idx, row_idx + rowspan):
                if r not in occupied:
                    occupied[r] = {}
                for c in range(col_idx, col_idx + colspan):
                    occupied[r][c] = True

            col_idx += colspan

        # 该行的有效列数为已占用的最大列索引+1
        if occupied[row_idx]:
            row_effective_cols[row_idx] = max(occupied[row_idx].keys()) + 1
        else:
            row_effective_cols[row_idx] = 0

    return row_effective_cols


def calculate_row_effective_columns(soup, row_idx):
    """计算指定行的有效列数（考虑rowspan占用）

    Args:
        soup: BeautifulSoup解析的表格
        row_idx: 行索引

    Returns:
        int: 该行的有效列数
    """
    row_effective_cols = build_table_occupied_matrix(soup)
    return row_effective_cols.get(row_idx, 0)


def calculate_row_columns(row):
    """
    计算表格行的实际列数，考虑colspan属性

    Args:
        row: BeautifulSoup的tr元素对象

    Returns:
        int: 行的实际列数
    """
    cells = row.find_all(["td", "th"])
    column_count = 0

    for cell in cells:
        colspan = int(cell.get("colspan", 1))
        column_count += colspan

    return column_count


def calculate_visual_columns(row):
    """
    计算表格行的视觉列数（实际td/th单元格数量，不考虑colspan）

    Args:
        row: BeautifulSoup的tr元素对象

    Returns:
        int: 行的视觉列数（实际单元格数）
    """
    cells = row.find_all(["td", "th"])
    return len(cells)


def detect_table_headers(soup1, soup2, max_header_rows=5):
    """
    检测并比较两个表格的表头

    Args:
        soup1: 第一个表格的BeautifulSoup对象
        soup2: 第二个表格的BeautifulSoup对象
        max_header_rows: 最大可能的表头行数

    Returns:
        tuple: (表头行数, 表头是否一致, 表头文本列表)
    """
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")

    # 构建两个表格的有效列数矩阵
    effective_cols1 = build_table_occupied_matrix(soup1)
    effective_cols2 = build_table_occupied_matrix(soup2)

    min_rows = min(len(rows1), len(rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for i in range(min_rows):
        # 提取当前行的所有单元格
        cells1 = rows1[i].find_all(["td", "th"])
        cells2 = rows2[i].find_all(["td", "th"])

        # 检查两行的结构和内容是否一致
        structure_match = True

        # 首先检查单元格数量
        if len(cells1) != len(cells2):
            structure_match = False
        else:
            # 检查有效列数是否一致（考虑rowspan影响）
            if effective_cols1.get(i, 0) != effective_cols2.get(i, 0):
                structure_match = False
            else:
                # 然后检查单元格的属性和内容
                for cell1, cell2 in zip(cells1, cells2):
                    colspan1 = int(cell1.get("colspan", 1))
                    rowspan1 = int(cell1.get("rowspan", 1))
                    colspan2 = int(cell2.get("colspan", 1))
                    rowspan2 = int(cell2.get("rowspan", 1))

                    # 去除所有空白字符（包括空格、换行、制表符等）
                    text1 = ''.join(full_to_half(cell1.get_text()).split())
                    text2 = ''.join(full_to_half(cell2.get_text()).split())

                    if colspan1 != colspan2 or rowspan1 != rowspan2 or text1 != text2:
                        structure_match = False
                        break

        if structure_match:
            header_rows += 1
            row_texts = [full_to_half(cell.get_text().strip()) for cell in cells1]
            header_texts.append(row_texts)  # 添加表头文本
        else:
            headers_match = header_rows > 0  # 只有当至少匹配了一行时，才认为表头匹配
            break

    # 如果严格匹配失败，尝试视觉一致性匹配（只比较文本内容）
    if header_rows == 0:
        header_rows, headers_match, header_texts = _detect_table_headers_visual(soup1, soup2, rows1, rows2, max_header_rows)

    return header_rows, headers_match, header_texts


def _detect_table_headers_visual(soup1, soup2, rows1, rows2, max_header_rows=5):
    """
    基于视觉一致性检测表头（只比较文本内容，忽略colspan/rowspan差异）

    Args:
        soup1: 第一个表格的BeautifulSoup对象
        soup2: 第二个表格的BeautifulSoup对象
        rows1: 第一个表格的行列表
        rows2: 第二个表格的行列表
        max_header_rows: 最大可能的表头行数

    Returns:
        tuple: (表头行数, 表头是否一致, 表头文本列表)
    """
    # 构建两个表格的有效列数矩阵
    effective_cols1 = build_table_occupied_matrix(soup1)
    effective_cols2 = build_table_occupied_matrix(soup2)

    min_rows = min(len(rows1), len(rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for i in range(min_rows):
        cells1 = rows1[i].find_all(["td", "th"])
        cells2 = rows2[i].find_all(["td", "th"])

        # 提取每行的文本内容列表（去除空白字符）
        texts1 = [''.join(full_to_half(cell.get_text()).split()) for cell in cells1]
        texts2 = [''.join(full_to_half(cell.get_text()).split()) for cell in cells2]

        # 检查视觉一致性：文本内容完全相同，且有效列数一致
        effective_cols_match = effective_cols1.get(i, 0) == effective_cols2.get(i, 0)
        if texts1 == texts2 and effective_cols_match:
            header_rows += 1
            row_texts = [full_to_half(cell.get_text().strip()) for cell in cells1]
            header_texts.append(row_texts)
        else:
            headers_match = header_rows > 0
            break

    if header_rows == 0:
        headers_match = False

    return header_rows, headers_match, header_texts


def get_neighbor_caption(blocks: List[Dict], table_idx: int) -> str:
    """获取表格紧邻上方的标题（用于逻辑判定）"""
    if table_idx <= 0:
        return ""

    # 获取前一个 block
    prev_block = blocks[table_idx - 1]
    b_type = prev_block.get("type", "")
    content = merge_para_with_text(prev_block).strip()

    # 1. 最确定的类型：table_caption
    if b_type == BlockType.TABLE_CAPTION:
        return content

    # 2. 其次：title (很多时候标题被识别为 title)
    if b_type == "title":
        return content

    # 3. 兜底：text (需要判断是否像标题)
    if b_type == BlockType.TEXT and len(content) < 150:
        # 匹配：Table 1, 表1, Exhibit A, 图2, 1.2 统计表, 三、情况说明
        pattern = r'^(table|表|exhibit|figure|图)\s*\d+|^\d+(\.\d+)*\s+|^[一二三四五六七八九十]+、'
        if re.match(pattern, content, re.IGNORECASE):
            return content

    return ""


def check_caption_consistency(page1_blocks: List[Dict], page2_blocks: List[Dict]) -> Tuple[bool, str]:
    """检查两个表格的标题一致性"""
    if not page1_blocks or not page2_blocks:
        return True, "页面Block为空，跳过一致性检查"

    # 1. 定位 Page 1 最后一个表格的索引
    p1_last_table_idx = -1
    for i in range(len(page1_blocks) - 1, -1, -1):
        if page1_blocks[i].get("type") == BlockType.TABLE:
            p1_last_table_idx = i
            break

    # 2. 定位 Page 2 第一个表格的索引
    p2_first_table_idx = -1
    for i in range(len(page2_blocks)):
        if page2_blocks[i].get("type") == BlockType.TABLE:
            p2_first_table_idx = i
            break

    if p1_last_table_idx == -1 or p2_first_table_idx == -1:
        return True, "无法在前后页中同时定位到表格，跳过Caption检查"

    # 3. 获取前一个Block作为潜在标题
    prev_caption = get_neighbor_caption(page1_blocks, p1_last_table_idx)
    curr_caption = get_neighbor_caption(page2_blocks, p2_first_table_idx)

    # Case 3: 均无标题 -> 一致
    if not prev_caption and not curr_caption:
        return True, "均无标题，判定为一致"

    # Case 1: 前表有，后表无 -> 一致 (视为续表)
    if prev_caption and not curr_caption:
        return True, f"前表有标题，后表无标题，视为续表"

    # Case 2: 前表无，后表有 -> 不一致 (视为新表)
    if not prev_caption and curr_caption:
        return False, f"前表无标题，后表有标题，视为新表"

    # Case 4: 均有标题 -> 详细比对 (编号)
    patterns = [
        (r'[Ee]xhibit\s*(\d+)', 'Exhibit'),
        (r'EXHIBIT\s*(\d+)', 'Exhibit'),
        (r'[Tt]able\s*([A-Za-z]?\s*-?\s*[\d.]+)', 'Table'),
        (r'[Tt]ab\.?\s*([A-Za-z]?\s*-?\s*[\d.]+)', 'Table'),
        (r'TABLE\s*([A-Za-z]?\s*-?\s*[\d.]+)', 'Table'),
        (r'表\s*([A-Za-z]?\s*-?\s*[\d.]+)', '表'),
        (r'[Ff]igure\s*(\d+)', 'Figure'),
        (r'图\s*(\d+)', '图'),
        (r'^([一二三四五六七八九十]+)、', 'CN_Num'),
    ]

    def extract_table_number(caption):
        for pattern, num_type in patterns:
            match = re.search(pattern, caption)
            if match:
                num_str = match.group(1).replace(' ', '').replace('-', '').upper()
                return num_type, num_str
        return None

    curr_res = extract_table_number(curr_caption)
    prev_res = extract_table_number(prev_caption)

    if curr_res and prev_res:
        prev_type, prev_num = prev_res
        curr_type, curr_num = curr_res
        if prev_type != curr_type or prev_num != curr_num:
            return False, f"表格编号不一致: {prev_type}{prev_num} vs {curr_type}{curr_num}"
        return True, f"表格编号一致: {prev_type}{prev_num}"

    # 文本比对
    def clean_caption(text):
        text = full_to_half(text).lower()
        for marker in CONTINUATION_MARKERS:
            if text.endswith(marker.lower()):
                text = text[: -len(marker)]
        text = re.sub(r'[^\w\u4e00-\u9fa5]', '', text)
        return text

    if clean_caption(prev_caption) != clean_caption(curr_caption):
        return False, "标题文本差异过大"

    return True, "标题一致"


def check_text_between_tables(page1_blocks: List[Dict], page2_blocks: List[Dict]) -> bool:
    """检查两个表格之间是否有正文内容阻断"""
    if not page1_blocks or not page2_blocks:
        return True

    # 检查第一页：最后一个表格后是否有 text
    last_table_idx = -1
    for i, block in enumerate(page1_blocks):
        if block.get("type") == BlockType.TABLE:
            last_table_idx = i

    if last_table_idx >= 0:
        for block in page1_blocks[last_table_idx + 1:]:
            if block.get("type") == BlockType.TEXT:
                return False

    # 检查第二页：第一个表格前是否有 text
    for block in page2_blocks:
        if block.get("type") == BlockType.TABLE:
            break
        elif block.get("type") == BlockType.TEXT:
            return False

    return True


def can_merge_tables(current_table_block, previous_table_block, page_blocks=None, previous_page_blocks=None):
    """判断两个表格是否可以合并"""
    # 1. 检查表间正文阻断
    if page_blocks is not None and previous_page_blocks is not None:
        if not check_text_between_tables(previous_page_blocks, page_blocks):
            return False, None, None, None, None

    # 2. 检查标题一致性
    if page_blocks is not None and previous_page_blocks is not None:
        is_consistent, _ = check_caption_consistency(previous_page_blocks, page_blocks)
        if not is_consistent:
            return False, None, None, None, None

    # 3. 检查表格是否有caption和footnote
    # 计算previous_table_block中的footnote数量
    footnote_count = sum(1 for block in previous_table_block["blocks"] if block["type"] == BlockType.TABLE_FOOTNOTE)
    # 如果有TABLE_CAPTION类型的块,检查是否至少有一个以"(续)"结尾
    caption_blocks = [block for block in current_table_block["blocks"] if block["type"] == BlockType.TABLE_CAPTION]
    if caption_blocks:
        # 检查是否至少有一个caption包含续表标识
        has_continuation_marker = False
        for block in caption_blocks:
            caption_text = full_to_half(merge_para_with_text(block).strip()).lower()
            if (
                    any(caption_text.endswith(marker.lower()) for marker in CONTINUATION_END_MARKERS)
                    or any(marker.lower() in caption_text for marker in CONTINUATION_INLINE_MARKERS)
            ):
                has_continuation_marker = True
                break

        # 如果所有caption都不包含续表标识，则不允许合并
        if not has_continuation_marker:
            return False, None, None, None, None

        # 如果current_table_block的caption存在续标识,放宽footnote的限制允许previous_table_block有最多一条footnote
        if footnote_count > 1:
            return False, None, None, None, None
    else:
        if footnote_count > 0:
            return False, None, None, None, None

    # 获取两个表格的HTML内容
    current_html = ""
    previous_html = ""

    for block in current_table_block["blocks"]:
        if (block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]):
            current_html = block["lines"][0]["spans"][0].get("html", "")

    for block in previous_table_block["blocks"]:
        if (block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]):
            previous_html = block["lines"][0]["spans"][0].get("html", "")

    if not current_html or not previous_html:
        return False, None, None, None, None

    # 检查表格宽度差异
    x0_t1, y0_t1, x1_t1, y1_t1 = current_table_block["bbox"]
    x0_t2, y0_t2, x1_t2, y1_t2 = previous_table_block["bbox"]
    table1_width = x1_t1 - x0_t1
    table2_width = x1_t2 - x0_t2

    if abs(table1_width - table2_width) / min(table1_width, table2_width) >= 0.1:
        return False, None, None, None, None

    # 解析HTML并检查表格结构
    soup1 = BeautifulSoup(previous_html, "html.parser")
    soup2 = BeautifulSoup(current_html, "html.parser")

    # 检查整体列数匹配
    table_cols1 = calculate_table_total_columns(soup1)
    table_cols2 = calculate_table_total_columns(soup2)
    # logger.debug(f"Table columns - Previous: {table_cols1}, Current: {table_cols2}")
    tables_match = table_cols1 == table_cols2

    # 检查首末行列数匹配
    rows_match = check_rows_match(soup1, soup2)

    return (tables_match or rows_match), soup1, soup2, current_html, previous_html


def check_rows_match(soup1, soup2):
    """检查表格行是否匹配"""
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")

    if not (rows1 and rows2):
        return False

    # 获取第一个表的最后一行数据行索引
    last_row_idx = None
    last_row = None
    for idx in range(len(rows1) - 1, -1, -1):
        if rows1[idx].find_all(["td", "th"]):
            last_row_idx = idx
            last_row = rows1[idx]
            break

    # 检测表头行数，以便获取第二个表的首个数据行
    header_count, _, _ = detect_table_headers(soup1, soup2)

    # 获取第二个表的首个数据行
    first_data_row_idx = None
    first_data_row = None
    if len(rows2) > header_count:
        first_data_row_idx = header_count
        first_data_row = rows2[header_count]  # 第一个非表头行

    if not (last_row and first_data_row):
        return False

    # 计算有效列数（考虑rowspan和colspan）
    last_row_effective_cols = calculate_row_effective_columns(soup1, last_row_idx)
    first_row_effective_cols = calculate_row_effective_columns(soup2, first_data_row_idx)

    # 计算实际列数（仅考虑colspan）和视觉列数
    last_row_cols = calculate_row_columns(last_row)
    first_row_cols = calculate_row_columns(first_data_row)
    last_row_visual_cols = calculate_visual_columns(last_row)
    first_row_visual_cols = calculate_visual_columns(first_data_row)

    # logger.debug(f"行列数 - 前表最后一行: {last_row_cols}(有效列数:{last_row_effective_cols}, 视觉列数:{last_row_visual_cols}), 当前表首行: {first_row_cols}(有效列数:{first_row_effective_cols}, 视觉列数:{first_row_visual_cols})")

    # 同时考虑有效列数匹配、实际列数匹配和视觉列数匹配
    return (last_row_effective_cols == first_row_effective_cols or
            last_row_cols == first_row_cols or
            last_row_visual_cols == first_row_visual_cols)


def check_row_columns_match(row1, row2):
    # 逐个cell检测colspan属性是否一致
    cells1 = row1.find_all(["td", "th"])
    cells2 = row2.find_all(["td", "th"])
    if len(cells1) != len(cells2):
        return False
    for cell1, cell2 in zip(cells1, cells2):
        colspan1 = int(cell1.get("colspan", 1))
        colspan2 = int(cell2.get("colspan", 1))
        if colspan1 != colspan2:
            return False
    return True


def adjust_table_rows_colspan(soup, rows, start_idx, end_idx,
                              reference_structure, reference_visual_cols,
                              target_cols, current_cols, reference_row):
    """调整表格行的colspan属性以匹配目标列数

    Args:
        soup: BeautifulSoup解析的表格对象（用于计算有效列数）
        rows: 表格行列表
        start_idx: 起始行索引
        end_idx: 结束行索引（不包含）
        reference_structure: 参考行的colspan结构列表
        reference_visual_cols: 参考行的视觉列数
        target_cols: 目标总列数
        current_cols: 当前总列数
        reference_row: 参考行对象
    """
    reference_row_copy = deepcopy(reference_row)

    # 构建有效列数矩阵
    effective_cols_matrix = build_table_occupied_matrix(soup)

    for i in range(start_idx, end_idx):
        row = rows[i]
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        # 使用有效列数（考虑rowspan）判断是否需要调整
        current_row_effective_cols = effective_cols_matrix.get(i, 0)
        current_row_cols = calculate_row_columns(row)

        # 如果有效列数或实际列数已经达到目标，则跳过
        if current_row_effective_cols >= target_cols or current_row_cols >= target_cols:
            continue

        # 检查是否与参考行结构匹配
        if calculate_visual_columns(row) == reference_visual_cols and check_row_columns_match(row, reference_row_copy):
            # 尝试应用参考结构
            if len(cells) <= len(reference_structure):
                for j, cell in enumerate(cells):
                    if j < len(reference_structure) and reference_structure[j] > 1:
                        cell["colspan"] = str(reference_structure[j])
        else:
            # 扩展最后一个单元格以填补列数差异
            # 使用有效列数来计算差异
            cols_diff = target_cols - current_row_effective_cols
            if cols_diff > 0:
                last_cell = cells[-1]
                current_last_span = int(last_cell.get("colspan", 1))
                last_cell["colspan"] = str(current_last_span + cols_diff)


def perform_table_merge(soup1, soup2, previous_table_block, wait_merge_table_footnotes):
    """执行表格合并操作"""
    # 检测表头有几行，并确认表头内容是否一致
    header_count, headers_match, header_texts = detect_table_headers(soup1, soup2)
    # logger.debug(f"检测到表头行数: {header_count}, 表头匹配: {headers_match}")
    # logger.debug(f"表头内容: {header_texts}")

    # 找到第一个表格的tbody，如果没有则查找table元素
    tbody1 = soup1.find("tbody") or soup1.find("table")

    # 获取表1和表2的所有行
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")


    if rows1 and rows2 and header_count < len(rows2):
        # 获取表1最后一行和表2第一个非表头行
        last_row1 = rows1[-1]
        first_data_row2 = rows2[header_count]

        # 计算表格总列数
        table_cols1 = calculate_table_total_columns(soup1)
        table_cols2 = calculate_table_total_columns(soup2)
        if table_cols1 >= table_cols2:
            reference_structure = [int(cell.get("colspan", 1)) for cell in last_row1.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(last_row1)
            # 以表1的最后一行为参考，调整表2的行
            adjust_table_rows_colspan(
                soup2, rows2, header_count, len(rows2),
                reference_structure, reference_visual_cols,
                table_cols1, table_cols2, first_data_row2
            )

        else:  # table_cols2 > table_cols1
            reference_structure = [int(cell.get("colspan", 1)) for cell in first_data_row2.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(first_data_row2)
            # 以表2的第一个数据行为参考，调整表1的行
            adjust_table_rows_colspan(
                soup1, rows1, 0, len(rows1),
                reference_structure, reference_visual_cols,
                table_cols2, table_cols1, last_row1
            )

    # 将第二个表格的行添加到第一个表格中
    if tbody1:
        tbody2 = soup2.find("tbody") or soup2.find("table")
        if tbody2:
            # 将第二个表格的行添加到第一个表格中（跳过表头行）
            for row in rows2[header_count:]:
                row.extract()
                tbody1.append(row)

    # 清空previous_table_block的footnote
    previous_table_block["blocks"] = [
        block for block in previous_table_block["blocks"]
        if block["type"] != BlockType.TABLE_FOOTNOTE
    ]
    # 添加待合并表格的footnote到前一个表格中
    for table_footnote in wait_merge_table_footnotes:
        temp_table_footnote = table_footnote.copy()
        temp_table_footnote[SplitFlag.CROSS_PAGE] = True
        previous_table_block["blocks"].append(temp_table_footnote)

    return str(soup1)


def merge_table(page_info_list):
    """合并跨页表格"""
    # 倒序遍历每一页
    for page_idx in range(len(page_info_list) - 1, -1, -1):
        # 跳过第一页，因为它没有前一页
        if page_idx == 0:
            continue

        page_info = page_info_list[page_idx]
        previous_page_info = page_info_list[page_idx - 1]

        # 检查当前页是否有表格块
        if not (page_info["para_blocks"] and page_info["para_blocks"][0]["type"] == BlockType.TABLE):
            continue

        current_table_block = page_info["para_blocks"][0]

        # 检查上一页是否有表格块
        if not (previous_page_info["para_blocks"] and previous_page_info["para_blocks"][-1]["type"] == BlockType.TABLE):
            continue

        previous_table_block = previous_page_info["para_blocks"][-1]

        # 收集待合并表格的footnote
        wait_merge_table_footnotes = [
            block for block in current_table_block["blocks"]
            if block["type"] == BlockType.TABLE_FOOTNOTE
        ]

        # 检查两个表格是否可以合并
        can_merge, soup1, soup2, current_html, previous_html = can_merge_tables(
            current_table_block, previous_table_block
        )

        if not can_merge:
            continue

        # 执行表格合并
        merged_html = perform_table_merge(
            soup1, soup2, previous_table_block, wait_merge_table_footnotes
        )

        # 更新previous_table_block的html
        for block in previous_table_block["blocks"]:
            if (block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]):
                block["lines"][0]["spans"][0]["html"] = merged_html
                break

        # 删除当前页的table
        for block in current_table_block["blocks"]:
            block['lines'] = []
            block[SplitFlag.LINES_DELETED] = True

def perform_semantic_merge_table(soup1: BeautifulSoup, soup2: BeautifulSoup, cell_list: List[int]) -> str:
    """
    cell_list:
        [1,0,0,0] # 第0个单元格需要语义合并, 第1,2,3个单元格直接拼接
        [0,0,0,0] # 第0,1,2,3 直接拼接
        [] # 不合并

    output:
        merged_html 两个表格合并后的结果
    """
    header_count, headers_match, header_texts = detect_table_headers(soup1, soup2)

    tbody1 = soup1.find("tbody") or soup1.find("table")

    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")

    skip_first_data_row = False  # 默认不跳过

    if rows1 and rows2 and header_count < len(rows2):
        visual_last_row_cells = get_visual_last_row_cells(soup1)
        first_data_row2: Tag = rows2[header_count]

        visual_last_row_cells_with_span_info = [
            (cell, origin_row_idx, int(cell.get("colspan", 1)), int(cell.get("rowspan", 1)))
            for cell, origin_row_idx in visual_last_row_cells
        ]
        first_data_row2_with_span_info = [
            (cell, int(cell.get("colspan", 1)), int(cell.get("rowspan", 1)))
            for cell in first_data_row2.find_all(["td", "th"])
        ]

        table_cols1 = calculate_table_total_columns(soup1)
        table_cols2 = calculate_table_total_columns(soup2)

        last_row1: Tag = rows1[-1]

        if table_cols1 >= table_cols2:
            reference_structure = [
                int(cell.get("colspan", 1))
                for cell in last_row1.find_all(["td", "th"])
            ]
            reference_visual_cols = calculate_visual_columns(last_row1)

            adjust_table_rows_colspan(
                rows2, header_count, len(rows2),
                reference_structure, reference_visual_cols,
                table_cols1, table_cols2, first_data_row2
            )
        else:  # table_cols2 > table_cols1
            reference_structure = [
                int(cell.get("colspan", 1))
                for cell in first_data_row2.find_all(["td", "th"])
            ]
            reference_visual_cols = calculate_visual_columns(first_data_row2)

            adjust_table_rows_colspan(
                rows1, 0, len(rows1),
                reference_structure, reference_visual_cols,
                table_cols2, table_cols1, last_row1
            )

        # 在结构对齐之后，按 cell_list 做语义合并
        skip_first_data_row = _merge_row_cells_by_cell_list(visual_last_row_cells, first_data_row2, cell_list, len(rows1))

    # 语义合并
    if tbody1:
        tbody2 = soup2.find("tbody") or soup2.find("table")
        if tbody2:
            # 将第二个表格的行添加到第一个表格中（跳过表头行）
            # 如果 skip_first_data_row 为 True(所有都为1），还要跳过第一个数据行
            start_idx = header_count + 1 if skip_first_data_row else header_count
            for row in rows2[start_idx:]:
                row.extract()
                tbody1.append(row)
    return str(soup1)


def _merge_row_cells_by_cell_list(visual_last_row_cells, first_data_row2, cell_list, total_rows1) -> bool:
    """
    [0,0,0,0]
    [1,0,0,0]
    ...
    """
    cells2 = first_data_row2.find_all(["td", "th"])

    max_idx = min(len(visual_last_row_cells), len(cells2), len(cell_list))

    last_row_idx = total_rows1 - 1

    cells_to_remove = []

    for i in range(max_idx):
        if cell_list[i] == 0:
            # cell_list[i] == 0: 直接拼接
            continue

        if cell_list[i] != 1:
            continue

        # cell_list[i] == 1: 语义合并
        c1, origin_row_idx = visual_last_row_cells[i]
        c2 = cells2[i]

        text1 = c1.get_text(" ", strip=True)
        text2 = c2.get_text(" ", strip=True)

        merged_text_parts = []
        if text1:
            merged_text_parts.append(text1)
        if text2:
            merged_text_parts.append(text2)

        merged_text = "".join(merged_text_parts)

        # 清空 c1 原内容
        for child in list(c1.contents):
            child.extract()
        if merged_text:
            c1.string = merged_text

        # 处理 rowspan：需要考虑 c2 的 rowspan
        c2_rowspan = int(c2.get("rowspan", 1))
        if origin_row_idx < last_row_idx:
           # c1不在最后一行,c1_rowspan+=c2_row_span
           current_rowspan = int(c1.get("rowspan", 1))
            c1["rowspan"] = str(current_rowspan + c2_rowspan)
            cells_to_remove.append(c2)
        elif c2_rowspan > 1:
            
            # c1 在最后一行，但 c2 有 rowspan，需要设置 c1 的 rowspan
            c1["rowspan"] = str(c2_rowspan)
            # 同样需要删除 c2，因为 c1 的 rowspan 覆盖了它
            cells_to_remove.append(c2)
        else:
            # c1 在最后一行且 c2 没有 rowspan，只需清空 c2 内容
            for child in list(c2.contents):
                child.extract()

    # 删除被 rowspan 覆盖的单元格
    for cell in cells_to_remove:
        cell.decompose()

    # 重新获取 cells2（因为可能有单元格被删除）
    cells2_remaining = first_data_row2.find_all(["td", "th"])

    # 检查 first_data_row2 是否所有单元格都被清空或删除了
    # 如果没有剩余单元格，或者剩余单元格都为空，应该跳过这一行
    if len(cells2_remaining) == 0:
        return True

    all_cells_empty = all(
        not c.get_text(strip=True)
        for c in cells2_remaining
    )

    if all_cells_empty:
        return True

    return False
