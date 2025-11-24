# Generate Question Answer Pairs from Markdown files
# These test cases are used by promptfoo to evaluate the RAG system.
import tiktoken
import os
import click
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import pandas as pd
from bickford.config import PROJECT_ROOT
from dotenv import load_dotenv
load_dotenv()

# Paths relative to project root
TESTS_DIR = PROJECT_ROOT / "tests"

SYSTEM_PROMPT = """
You are an assistant that generates high-quality evaluation questions for a Retrieval-Augmented Generation (RAG) question-answering system.

You are given a single document. Your job is to create question–answer pairs that:
- Are **strictly grounded** in the given document (no outside knowledge).
- Are **specific**, not generic (avoid “What is the purpose of this document?”).
- Help evaluate whether a RAG system has **retrieved the right context** and **answered faithfully**.

You MUST respond **only** in JSON, with this exact schema:

{
  "file": "<string: filename>",
  "pairs": [
    {
      "question": "<string: question>",
      "answer": "<string: concise, fully correct answer>"
    }
  ]
}

Rules:
- Generate **exactly 3** questions per document, unless the content is clearly too short to support that many.
- Every question **must** be answerable using ONLY the provided document.
- Do **not** include explanations, notes, or any text outside the JSON object.
"""


USER_PROMPT = """
You are given a single document.

Filename: "{file}"

Document content:
---
{file_content}
---

Your task:

1. Read the document carefully.
2. Generate 3 question–answer pairs that are useful for evaluating a RAG QA system.

Question design guidelines:
- Questions **must be specific** and grounded in details from the document.
- Avoid generic questions such as:
  - "What is the purpose of this document?"
  - "Summarize this document."
- Prefer questions that:
  - Focus on **concrete rules, conditions, exceptions, or procedures** described in the document.
  - Sometimes require integrating information from **two or more different parts** of the document (multi-hop).
  - Include at least one question that involves **numbers, thresholds, dates, or limits** (if available).
  - Include at least one question about **conditions or procedures** ("Under what conditions…", "What must a vessel do when…").
- Make sure each answer:
  - Can be answered **fully and correctly** using only this document.
  - Is phrased clearly and concisely.
  - Uses terminology consistent with the document itself.

Output:
- Follow the JSON schema defined in the system prompt.
- Use the provided filename "{file}" in the `file` field.
- Do not add any extra fields.
- Do not include any text before or after the JSON.
- Only include information from the source document.
"""

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

class QuestionAnswerPair(BaseModel):
    question: str
    answer: str

class QuestionAnswerPairs(BaseModel):
    pairs: List[QuestionAnswerPair]

def get_token_count(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

def generate_question_answer_pairs(file: Path, df: pd.DataFrame):
    """Generate question-answer pairs from a markdown file and add them to the DataFrame."""
    if file.suffix.lower() != ".md":
        print(f"Skipping {file} because it is not a markdown file")
        return df

    file_content = file.read_text()
    if get_token_count(file_content) > 100000:
        file_content = file_content[:100000]

    prompt = USER_PROMPT.format(file=file.name, file_content=file_content)
    response = client.beta.chat.completions.parse(
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}],
        response_format=QuestionAnswerPairs,
    )

    parsed_data = response.choices[0].message.parsed
    mdfile = file.name
    rows = []
    for pair in parsed_data.pairs:
        rows.append({
            "input": pair.question,
            "__expected1": f"llm-rubric: {pair.answer}",
            "__expected2": f"contains: [file={mdfile}]",
        })

    if rows:
        new_df = pd.DataFrame(rows)
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"Generated {len(rows)} test cases from {file.name}")

    return df


@click.command()
@click.argument("path", type=Path)
def generate_question_answer_pairs_from_file(path: Path):
    df = pd.DataFrame(columns=["input", "__prefix", "__expected", "__suffix"])

    if path.is_dir():
        for file in path.iterdir():
            try:
                df = generate_question_answer_pairs(file, df)
            except Exception as e:
                print(f"Error generating question-answer pairs from {file}: {e}")
                continue
    elif path.is_file() and path.suffix.lower() == ".md":
        df = generate_question_answer_pairs(path, df)
    else:
        print(f"Error: {path} is not a valid markdown file or directory")
        return

    # Write CSV file
    df.to_csv(TESTS_DIR / "cases.csv", index=False)
    print(f"Generated {len(df)} test cases and saved to {TESTS_DIR / 'cases.csv'}")

if __name__ == "__main__":
    generate_question_answer_pairs_from_file()