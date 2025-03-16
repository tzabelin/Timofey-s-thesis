import re

log_file = "outputs"
output_file = "data.dat"

log_pattern = re.compile(r"\[(\d{2}-\d{2}-\d{2})\] Rank \d+.*?my counter=(\d+)")

data_points = []

with open(log_file, "r") as file:
    for line in file:
        match = log_pattern.search(line)
        if match:
            time = match.group(1)
            counter = int(match.group(2))
            real_steps = counter / 2
            data_points.append((time, real_steps))

with open(output_file, "w") as file:
    for time, steps in data_points:
        file.write(f"{time} {steps}\n")
