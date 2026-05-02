import pandas as pd
import re
import sys

# ── 标点集 ──────────────────────────────────────────────────────────────────
PUNCT = set('，。？！；：""''（）【】《》、…—～·,.?!;:\'"()[]{}<>')

VALID_POS = {'N','V','A','VN','M','Q','R','D','P','C','SP','AS','Y','FW','ND','I','O','IDM','PU','X'}

# ── 加载词典 ─────────────────────────────────────────────────────────────────
def load_dict(path):
    df = pd.read_excel(path, header=None)
    df.columns = ['word', 'pos', 'definition'] + list(df.columns[3:])
    lexicon = {}  # word -> [pos, ...]
    for _, row in df.iterrows():
        word = str(row['word']).replace('|', '').strip()
        pos_raw = str(row['pos']).strip()
        if not word or word == 'nan':
            continue
        # 拆分复合词性 V+N -> ['V','N']
        tags = [t.strip() for t in re.split(r'[+/]', pos_raw) if t.strip()]
        tags = [t for t in tags if t in VALID_POS] or [pos_raw]
        if word not in lexicon:
            lexicon[word] = tags
        else:
            for t in tags:
                if t not in lexicon[word]:
                    lexicon[word].append(t)
    return lexicon

# ── 词性消歧 ─────────────────────────────────────────────────────────────────
def disambiguate(tags, prev_pos, next_pos):
    """从候选词性列表中选一个，基于前后词性做简单启发式。"""
    if len(tags) == 1:
        return tags[0]
    # 优先规则
    if 'FW' in tags: return 'FW'
    if 'ND' in tags: return 'ND'
    if 'IDM' in tags: return 'IDM'
    # V+N：前面是代词/名词则取V（谓语位），否则取第一个
    if 'V' in tags and 'N' in tags:
        return 'V' if prev_pos in ('R','N','VN',None) else 'N'
    if 'V' in tags and 'A' in tags:
        return 'V'
    if 'A' in tags and 'N' in tags:
        return 'N' if next_pos in ('Q','M') else 'A'
    return tags[0]

# ── 正向最大匹配 ──────────────────────────────────────────────────────────────
def fmm_segment(sentence, lexicon):
    tokens = []  # [(word, [tags])] 原始分词结果
    i = 0
    max_len = max((len(w) for w in lexicon), default=10)
    while i < len(sentence):
        ch = sentence[i]
        if ch in PUNCT:
            tokens.append((ch, ['PU']))
            i += 1
            continue
        matched = False
        for length in range(min(max_len, len(sentence)-i), 0, -1):
            candidate = sentence[i:i+length]
            if candidate in lexicon:
                tokens.append((candidate, lexicon[candidate]))
                i += length
                matched = True
                break
        if not matched:
            tokens.append((ch, ['X']))
            i += 1
    return tokens

# ── 后处理：SP合并、ND合并 ────────────────────────────────────────────────────
def postprocess(tokens):
    result = []
    i = 0
    while i < len(tokens):
        word, tags = tokens[i]
        # ND：名词/代词 + 头里/头上/头下 → 合并为 ND
        if i + 1 < len(tokens):
            nw, nt = tokens[i+1]
            if nw in ('头里','头上','头下','里向','外向') and tags[0] in ('N','R','VN'):
                result.append((word + nw, ['ND']))
                i += 2
                continue
        # SP黏着：当前词是SP，与前词合并
        if tags[0] == 'SP' and result:
            prev_word, prev_tags = result[-1]
            result[-1] = (prev_word + word, prev_tags)
            i += 1
            continue
        result.append((word, tags))
        i += 1
    return result

# ── 最终词性选择 ──────────────────────────────────────────────────────────────
def assign_pos(tokens):
    result = []
    for idx, (word, tags) in enumerate(tokens):
        prev_pos = result[-1][1] if result else None
        next_tags = tokens[idx+1][1] if idx+1 < len(tokens) else [None]
        pos = disambiguate(tags, prev_pos, next_tags[0])
        result.append((word, pos))
    return result

# ── 格式化一句 ────────────────────────────────────────────────────────────────
def format_sentence(annotated):
    return '#'.join(f"{w}/{p}" for w, p in annotated)

# ── 持久化写入词典文件 ────────────────────────────────────────────────────────
def save_to_dict(dict_path, word, pos):
    df = pd.read_excel(dict_path, header=None)
    new_row = pd.DataFrame([[word, pos, '']], columns=df.columns[:3] if len(df.columns) >= 3 else [0, 1, 2])
    # 补齐列数
    for col in df.columns[3:]:
        new_row[col] = ''
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(dict_path, index=False, header=False)
    print(f'  已写入 rare2oo.xlsx：{word} → {pos}')

# ── 交互式 X 确认 ─────────────────────────────────────────────────────────────
def interactive_confirm(sentence, annotated, lexicon, dict_path):
    """
    如果句子中有 X 词，暂停让用户确认。
    返回修正后的 annotated 列表，以及是否要继续运行。
    """
    x_words = [(i, w, p) for i, (w, p) in enumerate(annotated) if p == 'X']
    if not x_words:
        return annotated, True

    print('\n' + '='*60)
    print(f'句子：{sentence}')
    print('当前标注：')
    print(format_sentence(annotated))
    print()
    print('未识别词（标为 X）：')
    for idx, w, p in x_words:
        print(f'  [{idx}] {w}')
    print()
    print('请反馈（直接回车跳过，输入 q 退出程序）：')
    print('  索引 词性        如 "0 N" 将第0个X词改为N')
    print('  索引 词性 add    改词性并写入词典')
    print('  rewrite 手臂/N#巴酸得/A#...  整句重写（覆盖全部标注）')
    print('  add 词 词性      将指定词写入词典')
    print('  re               用当前词典重新自动分词（加词后使用）')

    while True:
        line = input('> ').strip()
        if not line:
            break
        if line.lower() == 'q':
            return annotated, False
        if line.lower() == 're':
            raw_tokens = fmm_segment(sentence, lexicon)
            processed = postprocess(raw_tokens)
            annotated = assign_pos(processed)
            x_words = [(i, w, p) for i, (w, p) in enumerate(annotated) if p == 'X']
            print(f'  重新分词结果：{format_sentence(annotated)}')
            if x_words:
                print('  仍有未识别词：')
                for i, w, _ in x_words:
                    print(f'    [{i}] {w}')
            continue
        parts = line.split(maxsplit=1)

        # add 词 词性
        if parts[0] == 'add':
            sub = parts[1].split() if len(parts) > 1 else []
            if len(sub) >= 2:
                new_word, new_pos = sub[0], sub[1]
                lexicon[new_word] = [new_pos]
                save_to_dict(dict_path, new_word, new_pos)
            else:
                print('  格式：add 词 词性')
            continue

        # rewrite 整句重写
        if parts[0] == 'rewrite' and len(parts) > 1:
            new_tokens = []
            for seg in parts[1].split('#'):
                seg = seg.strip()
                if '/' in seg:
                    sw, sp = seg.rsplit('/', 1)
                    new_tokens.append((sw, sp.strip()))
                else:
                    new_tokens.append((seg, 'X'))
            annotated = new_tokens
            print(f'  整句已重写：{format_sentence(annotated)}')
            continue

        # 索引 词性 [add]
        tokens_line = line.split()
        if len(tokens_line) >= 2 and tokens_line[0].isdigit():
            target_idx = int(tokens_line[0])
            new_pos = tokens_line[1]
            do_add = len(tokens_line) >= 3 and tokens_line[2] == 'add'
            x_list = [i for i, (w, p) in enumerate(annotated) if p == 'X']
            if target_idx < len(x_list):
                x_pos = x_list[target_idx]
                word = annotated[x_pos][0]
                annotated[x_pos] = (word, new_pos)
                print(f'  已修改：{word} → {new_pos}')
                if do_add:
                    lexicon[word] = [new_pos]
                    save_to_dict(dict_path, word, new_pos)
            else:
                print(f'  索引 {target_idx} 超出范围')
            continue

        print('  未识别指令，请重试')

    print('修正后标注：')
    print(format_sentence(annotated))
    print()
    cont = input('继续下一句？(回车继续 / q 退出) > ').strip()
    return annotated, cont.lower() != 'q'

# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    dict_path = 'rare2oo.xlsx'
    input_path = 'output.xlsx'
    output_path = 'output.xlsx'

    print('加载词典...')
    lexicon = load_dict(dict_path)
    print(f'词典加载完成，共 {len(lexicon)} 条词条')

    df = pd.read_excel(input_path)
    if 'sentence' not in df.columns:
        df.columns = ['sentence'] + list(df.columns[1:])

    if 'annotation' not in df.columns:
        df['annotation'] = ''

    print(f'共 {len(df)} 个句子待标注\n')

    for idx, row in df.iterrows():
        sentence = str(row['sentence']).strip()
        if not sentence or sentence == 'nan':
            continue
        # 已标注则跳过
        if pd.notna(row.get('annotation')) and str(row.get('annotation','')):
            continue

        # 分词 + 后处理 + 词性赋值
        raw_tokens = fmm_segment(sentence, lexicon)
        processed = postprocess(raw_tokens)
        annotated = assign_pos(processed)

        # 交互确认 X
        annotated, should_continue = interactive_confirm(sentence, annotated, lexicon, dict_path)

        # 写入
        df.at[idx, 'annotation'] = format_sentence(annotated)
        df.to_excel(output_path, index=False)

        if not should_continue:
            print('已保存，程序退出。')
            sys.exit(0)

    print('\n全部标注完成！结果已写入 output.xlsx')

if __name__ == '__main__':
    main()
