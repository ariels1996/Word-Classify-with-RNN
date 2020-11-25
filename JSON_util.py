import json
import csv

input_file_name = "result.txt"
output_file_name = "data1.csv"

with open(input_file_name, 'r', encoding='utf-8', newline='') as input_file, \
    open(output_file_name, 'w', encoding='utf-8', newline='') as output_file:
    
    data = []
    for line in input_file:
        datum = json.loads(line)
        data.append(datum)

    csvwriter = csv.writer(output_file)
    csvwriter.writerow(["pattern","isVulnerable"])
    for line in data:
        if "superLinear" in line: 
            if "pattern" in line["superLinear"] and "isVulnerable" in line["superLinear"]:
                csvwriter.writerow([line["superLinear"]["pattern"],
                                    line["superLinear"]["isVulnerable"]
                                    ])
