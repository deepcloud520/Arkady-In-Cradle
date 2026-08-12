from pathlib import Path

def total_code_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        count = len(f.readlines())
    return count
if __name__ == "__main__":
    root_dir = Path(__file__).parent / "Airis"
    print(f"Counting code lines in {root_dir} ...")
    total_lines = 0
    for file_path in root_dir.rglob("*.py"):
        lines = total_code_lines(file_path)
        total_lines += lines
        print(f"{file_path.name}: {lines} lines")
    print(f"Total code lines: {total_lines}")