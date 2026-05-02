"""一次性脚本：给根目录 processed_results.csv 加上表头。

如果 CSV 正被 Excel 打开，关掉 Excel 后再运行。
"""

from pathlib import Path

HEADER = "entry,entry_alt,romanization,ipa,definition,tone_notation,notes\n"

def main() -> None:
    p = Path(__file__).resolve().parents[1] / "processed_results.csv"
    raw = p.read_bytes()
    has_bom = raw[:3] == b"\xef\xbb\xbf"
    prefix = b"\xef\xbb\xbf" if has_bom else b""
    body = raw[3:] if has_bom else raw

    # Check if header already exists
    first_line = body.split(b"\n")[0].decode("utf-8-sig", errors="replace")
    if first_line.startswith("entry,"):
        print("表头已存在，无需操作。")
        return

    new_content = prefix + HEADER.encode("utf-8") + body
    p.write_bytes(new_content)
    print(f"完成：已在 {p} 头部插入表头。")

if __name__ == "__main__":
    main()
