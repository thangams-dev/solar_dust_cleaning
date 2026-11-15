import csv
import io


def records_to_csv(records: list[dict]) -> str:
    if not records:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()
