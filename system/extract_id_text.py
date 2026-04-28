import json
from pathlib import Path


def load_records(path: Path):
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    records = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def main():
    input_path = Path("test.json")
    output_path = Path("input.json")

    records = load_records(input_path)
    result = [
        {
            "id": record.get("id"),
            "text": record.get("text"),
        }
        for record in records
        if "id" in record or "text" in record
    ]

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(result)} records to {output_path}")


if __name__ == "__main__":
    main()
