import sys
import requests

# Enable debugging with True or disable it using False
DEBUG = "True"

# Download Tranco List (full file)
if DEBUG == "True":
    print("Downloading Tranco List")

tranco_url = "https://tranco-list.eu/download/3V62L/full"
r = requests.get(tranco_url, allow_redirects = True)
open("tranco_full_list.csv", "wb").write(r.content)

if DEBUG == "True":
    print("Tranco List was downloaded")

# Specify the file containing DGA names from DGArchive. Load them to a Python set.
dga_filename = "./dgarchive.csv"
all_dgas_set = set()

if DEBUG == "True":
    print("Reading DGArchive and loading names to a set")

counter = 0
with open(dga_filename) as infile:
    for line in infile:
        line = line.strip()
        parts = line.split(",")
        name = parts[0].replace('"', '')
        all_dgas_set.add(name)
        counter += 1
        if DEBUG == "True":
            if counter % 10000000 == 0:
                print("Processed Names: ", counter)

if DEBUG == "True":
    print("DGArchive names were loaded to a set")

# This file will contain the DGA names found within the Tranco List
fdw = open("dga_names_in_tranco.txt", "w")
# This file will be Tranco list without the DGA names, numbered as a CSV
fdw2 = open("tranco_full_original.csv", "w")

tranco_filename = "tranco_full_list.csv"
removed_names = set()

if DEBUG == "True":
    print("Reading Tranco List to find DGA's within the list")

with open(tranco_filename) as infile:
    for line in infile:
        line2 = line.strip()
        no, name = line2.split(",")
        if name in all_dgas_set:
            removed_names.add(name)
            fdw.write(name + "\n")
        else:
            fdw2.write(line)

fdw.close()
fdw2.close()

if DEBUG == "True":
    print("DGA names withing Tranco were found and Tranco List was cleared")

# Read Tranco list and extract top 100K names for whitelist feature
fdw = open("tranco_top100k.txt", "w")
# Keep the remaining names in another file
fdw2 = open("tranco_remaining.txt", "w")

if DEBUG == "True":
    print("Starting to split Tranco into top 100k names and remaining ones")

counter = 0
with open("tranco_full_original.csv") as infile:
    for line in infile:
        counter += 1
        line = line.strip()
        no, name = line.split(",")
        if counter <= 100000:
            fdw.write(name + "\n")
        else:
            fdw2.write(name + "\n")

fdw.close()
fdw2.close()

if DEBUG == "True":
    print("Done separating Tranco to top 100k and remaining names")
