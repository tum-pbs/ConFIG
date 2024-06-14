#usr/bin/python3

#version:0.0.1
#last modified:20230803

import csv

def write_dictlist(dictlist,file_name):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dictlist[0].keys(),delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for line in dictlist:
            writer.writerow(line)

def read_dictlist(file_name): 
    with open(file_name,newline='') as csvfile:
        reader=csv.DictReader(csvfile,delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)
        return [row for row in reader]
