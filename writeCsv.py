#!/usr/bin/env python3

import csv


def writeCsvRow(writer, index, color, color_name, hex_code, r, g, b):
    writer.writerow({index[0]: color,
                    index[1]: color_name,
                    index[2]: hex_code,
                    index[3]: r,
                    index[4]: g,
                    index[5]: b})



with open('basic_colors_simplified.csv', mode='w') as csv_file:
    index=["color","color_name","hex","R","G","B"]
    writer = csv.DictWriter(csv_file, fieldnames=index)
    # writer.writeheader()
    
    # Basic Colors
    # writeCsvRow(writer, index, "white", "white", "#FFFFFF", "255", "255", "255")
    # writeCsvRow(writer, index, "silver", "silver", "#C0C0C0", "192", "192", "192")
    # writeCsvRow(writer, index, "gray", "gray", "#808080", "128", "128", "128")
    # writeCsvRow(writer, index, "black", "black", "#000000", "0", "0", "0")
    # writeCsvRow(writer, index, "red", "red", "#FF0000", "255", "0", "0")
    # writeCsvRow(writer, index, "maroon", "maroon", "#800000", "128", "0", "0")
    # writeCsvRow(writer, index, "yellow", "yellow", "#FFFF00", "255", "255", "0")
    # writeCsvRow(writer, index, "olive", "olive", "#808000", "128", "128", "0")
    # writeCsvRow(writer, index, "lime", "lime", "#00FF00", "0", "255", "0")
    # writeCsvRow(writer, index, "green", "green", "#008000", "0", "128", "0")
    # writeCsvRow(writer, index, "aqua", "aqua", "#00FFFF", "0", "255", "255")
    # writeCsvRow(writer, index, "teal", "teal", "#008080", "0", "128", "128")
    # writeCsvRow(writer, index, "blue", "blue", "#0000FF", "0", "0", "255")
    # writeCsvRow(writer, index, "navy", "navy", "#000080", "0", "0", "128")
    # writeCsvRow(writer, index, "fuchsia", "fuchsia", "#FF00FF", "255", "0", "255")
    # writeCsvRow(writer, index, "purple", "purple", "#800080", "128", "0", "128")
    
    # Basic Colors Simplified
    writeCsvRow(writer, index, "white", "white", "#FFFFFF", "255", "255", "255")
    writeCsvRow(writer, index, "gray", "gray", "#C0C0C0", "192", "192", "192")
    writeCsvRow(writer, index, "gray", "gray", "#808080", "128", "128", "128")
    writeCsvRow(writer, index, "black", "black", "#000000", "0", "0", "0")
    writeCsvRow(writer, index, "red", "red", "#FF0000", "255", "0", "0")
    writeCsvRow(writer, index, "red", "red", "#800000", "128", "0", "0")
    writeCsvRow(writer, index, "yellow", "yellow", "#FFFF00", "255", "255", "0")
    writeCsvRow(writer, index, "green", "green", "#808000", "128", "128", "0")
    writeCsvRow(writer, index, "green", "green", "#00FF00", "0", "255", "0")
    writeCsvRow(writer, index, "green", "green", "#008000", "0", "128", "0")
    writeCsvRow(writer, index, "blue", "blue", "#00FFFF", "0", "255", "255")
    writeCsvRow(writer, index, "green", "green", "#008080", "0", "128", "128")
    writeCsvRow(writer, index, "blue", "blue", "#0000FF", "0", "0", "255")
    writeCsvRow(writer, index, "blue", "blue", "#000080", "0", "0", "128")
    writeCsvRow(writer, index, "purple", "purple", "#FF00FF", "255", "0", "255")
    writeCsvRow(writer, index, "purple", "purple", "#800080", "128", "0", "128")


    