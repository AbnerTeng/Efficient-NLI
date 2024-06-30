import json
from .utils import parse_json_file

if __name__ == "__main__":
    with open(
        "SNLI/snli_1.0_dev.jsonl",
        'r', encoding="utf-8"
    ) as json_file:
        data = json_file.readlines()
    new_data = parse_json_file(data)
    print(new_data[0])
