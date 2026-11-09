from pathlib import Path

def total_code_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        count = len(f.readlines())
    return count
def count_code_lines_in_directory(directory: Path, glob_pattern: str = "*.py") -> int:
    total_lines = 0
    for file_path in directory.rglob(glob_pattern):
        total_lines += total_code_lines(file_path)
    print(f"Total code lines in {directory}: {total_lines}")
    return total_lines
if __name__ == "__main__":
    root_dir = Path(__file__).parent 
    total = count_code_lines_in_directory(root_dir / "Airia")
    total += count_code_lines_in_directory(root_dir / "tests")
    total += count_code_lines_in_directory(root_dir / "docs", glob_pattern="*.md")
    print(f"Total code lines: {total}")