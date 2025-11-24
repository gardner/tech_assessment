# Convert PDF files to Markdown

from pathlib import Path
import pymupdf4llm
import click

def convert_file_with_cache(file: Path, to_path: Path):
    if to_path.exists():
        print(f"Skipping {file} because {to_path} already exists")
        return
    if file.is_file() and file.suffix.lower() == ".pdf":
        md_text = pymupdf4llm.to_markdown(file)
        with open(to_path, "w") as f:
            f.write(md_text)
        print(f"Converted {file} to {to_path}")
    else:
        print(f"Skipping: {file} is not a valid PDF file")


@click.command()
@click.argument("from_path", type=Path, default=Path("pdf"))
@click.argument("to_path", type=Path, default=Path("data"))
def main(from_path: Path, to_path: Path):
    if from_path.is_dir():
        for file in from_path.iterdir():
            from_file = Path(file)
            to_file = to_path / f"{from_file.stem}.md"
            if from_file.suffix.lower() == ".pdf":
                convert_file_with_cache(from_file, to_file)
            else:
                print(f"Skipping {file} because it is not a valid PDF file")
    elif from_path.is_file() and from_path.suffix.lower() == ".pdf":
        convert_file_with_cache(from_path, to_path / f"{from_path.stem}.md")
    else:
        print(f"Error: {from_path} is not a valid PDF file or directory")
        return

if __name__ == "__main__":
    main()