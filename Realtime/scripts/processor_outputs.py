import re
from datetime import datetime

log_lines = open("outputs_partial.txt").readlines()
rank0_data = []

time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}) - Rank (\d+) \(global_id 0\): counter=(\d+)")

start_time = None

for line in log_lines:
    match = time_pattern.search(line)
    if match:
        time_str, rank_str, counter = match.groups()
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        if start_time is None:
            start_time = time_obj
        elapsed_seconds = (time_obj - start_time).total_seconds()
        rank0_data.append((elapsed_seconds, int(counter) / 3))

with open("outputs", "w") as f:
    for t, val in rank0_data:
        f.write(f"{int(t)}\t{val:.2f}\n")
