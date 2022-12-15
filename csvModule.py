import csv

def openCsvFile(headerTitle, filename):
    header = [headerTitle]
    f = open(filename, 'w')
    # create the csv writer
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    return f, writer

def closeCsvFile(f):
    f.close()