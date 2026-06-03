import os
import torch
import json
from TTS.tts.models.vits import Vits, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer



from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.shared_configs import CharactersConfig
import re
import time
import numpy as np
from scipy.io import wavfile  # 替代 save_wav


# 务必确保这里的 WUU_MAP 和训练脚本中的完全一致
WUU_MAP = {
    # 使用亚美尼亚语和格鲁吉亚语小写字母，每个值唯一
    # 亚美尼亚语：ա բ գ դ ե զ է ը թ ժ ի լ խ ծ կ հ ձ ղ ճ մ յ ն շ ո չ պ ջ ռ ս վ տ ու փ ք օ ֆ
    # 格鲁吉亚语：ა ბ გ დ ე ვ ზ თ ი კ ლ მ ნ ო პ ჟ რ ს ტ უ ფ ქ ღ ყ შ ჩ ც ძ წ ჭ ხ ჯ ჰ

    # 1. 多字符声母
    "tsh": "ա", "gh": "բ", "gn": "գ", "ng": "դ", "ts": "ե",
    "ch": "զ", "sh": "է", "zh": "ը", "kh": "թ", "ph": "ժ", "th": "ի",

    # 2. 四字符韵母
    "iaon": "լ", "yaon": "խ", "uaon": "ծ", "waon": "կ",

    # 3. 三字符韵母
    "iau": "հ", "yau": "ձ", "ioe": "ղ", "yoe": "ճ", "uoe": "մ", "woe": "յ",
    "iun": "ն", "yun": "շ", "uen": "ո", "wen": "չ", "ian": "պ", "yan": "ջ",
    "uan": "ռ", "wan": "ս", "iaq": "վ", "yaq": "დ", "uaq": "ე", "waq": "ვ",
    "ueq": "ზ", "weq": "თ", "ioq": "ი", "yoq": "კ", "iuq": "ლ", "yuq": "მ", "yiq": "ნ",

    # 4. 二字符韵母
    "aon": "ო", "ia": "პ", "ya": "ჟ", "ua": "რ", "wa": "ს", "ie": "ტ", "ye": "უ",
    "ue": "ფ", "we": "ქ", "au": "ღ", "oe": "ყ", "iu": "შ", "yu": "ჩ", "in": "ც",
    "yin": "ძ", "en": "წ", "an": "ჭ", "on": "ხ", "ion": "ჯ", "yon": "ჰ", "iq": "ք",
    "aq": "օ", "eq": "ֆ", "oq": "ְ", "yi": "ֵ", "wu": "ַ", "er": "ָ", "eu": "ֹ",

    # 5. 其他韵母
    "aen": "ֺ", "een": "ּ", "oen": "ֽ", "eon": "ֿ", "aeq": "ס",
    "uoq": "ױ", "woq": "ײ", "uaeq": "א", "waeq": "ב",
    "yeu": "ד", "ieu": "ה",
    "eoq": "є", "oeq": "ѕ",
    "uon": "ѓ", "won": "ѝ",
    "io": "ֶ", "yo": "ֱ", "uo": "ֲ", "wo": "ֳ",
    "ioeq": "ю", "yoeq": "я", "uoeq": "ё", "woeq": "ђ",
    "iaen": "а", "yaen": "б", "uaen": "в", "waen": "г",
    "ien": "ж", "yen": "й",
    "ioen": "щ", "yoen": "ъ", "uoen": "ы", "woen": "ь",
}


initials_list = [
        "tsh", "gh", "gn", "ng", "ts", "ch", "sh", "zh", "kh", "ph", "th",
        "p", "b", "m", "f", "v", "t", "d", "n", "l", "s", "z", "c", "j", "k", "g", "h"
    ]
    # 2. 定义韵母表 (降序排列)
finals_list = [
        "iaon", "yaon", "uaon", "waon", 'iaen','yaen','uaen','waen','ien','yen', 
        'uen','wen','ioen','yoen','uoen','woen','ian','yan','uan','wan',
        'in','yin','ion','yon','uon','won','iaq','yaq','uaq','waq','uaeq','waeq',
        'iq','yiq','ueq','weq','ioeq','yoeq','uoeq','woeq','ioq','yoq','uoq','woq',
        'ieu','yeu','iau','yau',
        'iu','yu','ia','ya','ua','wa','ie','ye','ue','we','io','yo','uo','wo'
        'aen','een','oen','an','eon','aon','on','aq','aeq','eq','eoq','oeq','oq',
        'eu','au'
        'yi','wu','y','a','e','o','i','u','m','gn','ng','n','l'
    ]


def prepare_text(raw_text):
    # 1. 定义拼音零件列表 (用于正则匹配)
    
    
    # 2. 构造正则：最长匹配原则
    re_initials = f"({'|'.join(sorted(initials_list, key=len, reverse=True))})?"
    re_finals = f"({'|'.join(sorted(finals_list, key=len, reverse=True))})"
    re_tones = r"(\d+)"
    pattern = re.compile(re_initials + re_finals + re_tones)

    tokens = []
    # 按照空格处理每一个词（比如 "ngu23 teq5"）
    for word in raw_text.split():
        matches = pattern.findall(word)
        if matches:
            for m in matches:
                # m 是一个元组 (声母, 韵母, 声调)
                for part in m:
                    if part:
                        # 核心步骤：如果在映射表里就换掉，不在就原样保留（如声母 p, t 或声调数字）
                        tokens.append(WUU_MAP.get(part, part))
        else:
            # 如果没匹配上（比如标点），原样保留
            tokens.append(word)
            
    # 将转换后的“火星文”零件用空格连起来返回
    return " ".join(tokens)

# 3. 加载模型
MODEL_RUN_PATH = r"E:\shaoxing_tts\tts"
config = VitsConfig()
CONFIG_PATH = os.path.join(MODEL_RUN_PATH, "config.json")
# 直接用 json 读取，确保万无一失
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_dict = json.load(f)

# 初始化 VitsConfig 对象
config = VitsConfig()
config.from_dict(config_dict)

# --- 2. 强制覆盖字符配置 (无论加载进来的是什么) ---
print("正在手动强制注入字符集补丁...")

# 必须和训练脚本逻辑严格一致
base_chars = list("pbmfvtdnlszcjkhgywaeiouywaeiou") # 确保包含所有单字母
tones = ["1", "2", "3", "4", "5", "0"]
mapped_chars = [v.lower() for v in WUU_MAP.values()]

# 这里的排序(sorted)非常重要，必须和训练时生成的顺序一致
custom_symbols = sorted(list(set(base_chars + tones + mapped_chars)))

# 强制重新赋值
config.characters = CharactersConfig(
    characters=custom_symbols,
    punctuations=",.!?",
    pad="_",
    eos="~",
    blank=" ",
)

print(f"字符集注入成功，共有 {len(config.characters.characters)} 个符号。")

# --- 3. 初始化 Tokenizer ---
# 现在这里绝对不会再报 NoneType 了
tokenizer_tuple = TTSTokenizer.init_from_config(config)
if isinstance(tokenizer_tuple, tuple):
    tokenizer = tokenizer_tuple[0]
else:
    tokenizer = tokenizer_tuple
model = Vits.init_from_config(config, tokenizer)

# 加载权重
CHECKPOINT_PATH = os.path.join(MODEL_RUN_PATH, "checkpoint_27000.pth")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

# 兼容不同的 checkpoint 格式
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()
if torch.cuda.is_available():
    model.cuda()

print("模型完全加载成功，准备合成！")

def synthesize(text, output_filename):
    # 1. 文本预处理
    processed_text = prepare_text(text)
    print(f"输入文本: {text}")
    print(f"映射结果: {processed_text}")

    # 2. 转化为 ID (假设你已经初始化好了 tokenizer)
    # 注意: 如果你训练时用了 basic_cleaners，这里也要对应上
    seq = tokenizer.text_to_ids(processed_text)
    text_inputs = torch.IntTensor(seq).unsqueeze(0)
    
    if torch.cuda.is_available():
        text_inputs = text_inputs.cuda()

    # 3. 模型推理
    print("正在合成...")
    start_time = time.time()
    with torch.no_grad():
        # model 是你加载好的 VITS 模型
        output = model.inference(
            text_inputs,
            config,
            #speaker_id=0, # 如果是单人模型通常为 0
        )
    
    # 4. 获取音频波形 Tensor
    # output["model_outputs"] 通常是波形，根据你的 TTS 版本可能略有不同
    wav = output["model_outputs"].cpu().numpy().flatten()
    
    # 2. 采样率从 config 中获取
    sampling_rate = config.audio.sample_rate
    
    # 3. 使用 scipy 保存
    # 这里的 wav 通常在 -1 到 1 之间，scipy 可以直接保存 float32 格式的 wav
    wavfile.write(output_filename, sampling_rate, wav)
    
    print(f"完成！已生成: {output_filename}")
    end_time = time.time()
    print(f"成功保存到: {output_filename} (耗时: {end_time - start_time:.2f}s)")
if __name__ == '__main__':
# --- 3. 调用示例 ---
    test_pinyin = "shi335 yan55 kua55 chi52"
    synthesize(test_pinyin, "shanghai_test.wav")