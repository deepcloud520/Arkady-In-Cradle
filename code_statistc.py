from pathlib import Path

def total_code_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        count = len(f.readlines())
    return count
def count_code_lines_in_directory(directory: Path) -> int:
    print(f"Counting code lines in {directory} ...")
    total_lines = 0
    for file_path in directory.rglob("*.py"):
        total_lines += total_code_lines(file_path)
    return total_lines
if __name__ == "__main__":
    root_dir = Path(__file__).parent / "Airis"
    print(f"Total code lines: {count_code_lines_in_directory(root_dir)}")