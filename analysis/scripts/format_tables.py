#!/usr/bin/env python3
"""
Created on: 21/06/2023 15:11

Author: Shyam Bhuller

Description: To help deal with table glorp
"""
import argparse
import numpy as np

from rich import print

def split_rows(row : str) -> list[str]:
    """ Takes a line read from a latex table and splits the row wherever there is a & symbol,
        if the line has no & symbols, it returns an empty list so naturally strips unwanted lines.

    Args:
        row (str): line from latex file

    Returns:
        list[str]: split rows.
    """
    split_row = []
    last = -1
    for i in range(len(row)):
        if row[i] == "&":
            split_row.append(row[last + 1:i])
            last = i
        if i == len(row) - 1 and last != -1: # we are at the end of the row
            split_row.append(row[last + 1:i-2]) # i -3 to remove newline characters \\
    return split_row


def format(string : str) -> str:
    """ Formats strings by stripping any latex formatting, if numeric it will also convert the string to a float or int.

    Args:
        string (str): string to format

    Returns:
        str: formatted string
    """
    replacements = {
    "{}": "",
    " ": "",
    "\\$" : "$",
    "\\textbackslash" : "\\",
    "\\textasciicircum" : "^",
    "\\{" : "{",
    "\\}" : "}",
    "\\_" : "_",
    "\greater" : ">"
    }
    tmp = string
    for original, new in replacements.items():
        tmp = tmp.replace(original, new)
    if tmp.isdigit(): # only numbers
        tmp = int(tmp)
    elif tmp.replace(".", "").isdigit(): # only numbers and decimal points
        tmp = float(tmp)
    return tmp


def TablesToArrays(tables : list) -> list:
    """ Takes a set of Latex tables (line by line) and converts it to an array of the table elements, formatting the contents.

    Args:
        tables (list): set of tables

    Returns:
        list: set of table element arrays
    """
    split_tables = []
    for i in range(len(tables)):
        tmp = []
        for l in tables[i]:
            split = [format(s) for s in split_rows(format(l))] # call format on whole list to remove latex words, then individual elements to convert to ints, I know, they should be two different methods :)
            if len(split) > 0:
                tmp.append(np.transpose(np.array(split, dtype = object)))
        split_tables.append(tmp)
    return np.array(split_tables)


def MergeTableArrayElements(split_tables : np.array) -> tuple[np.array]:
    """ Takes a set of table element arrays and merges the elements such that each entry in the table is a tuple rather than a single entry i.e.
        [ A | B ], [ A | B ]         [   A   |   B   ]
        [ 0 | 1 ], [ 3 | 2 ] becomes [ (0,3) | (1,2) ].
        It will split the table into merged elements, row labels and columns labels

    Args:
        split_tables (np.array): set of table element arrays

    Returns:
        tuple[np.array]: merged elements, row labels, column labels.
    """
    merged_elements = []

    for i, t in enumerate(zip(*split_tables)):
        if i == 0: continue # row labels

        merged = np.vstack([t[j][1:] for j in range(len(t))]).T # dont merge column headers
        merged_elements.append(merged)

    column_headers = split_tables[0][0]
    row_labels = split_tables[0][:, 0]
    return np.array(merged_elements), row_labels, column_headers


def ReformatElements(elements : np.array) -> list[str]:
    """ Reformats the merged elements by first converting the object into a string (floats are rounded to 3 decimal places).

    Args:
        elements (np.array): elements of table

    Returns:
        list[str]: Latex formatted elements.
    """
    merged_elements_str = []
    for i in elements:
        row = []
        for j in i:
            string = ""
            for s in j:
                if type(s) == int or (type(s) == float and s.is_integer()):
                    string += f"& {int(s)} "
                else:
                    string += f"& {s:.3f} "
            row.append(string)
        merged_elements_str.append(row)
    return merged_elements_str


def MergeElementRows(elements_str : list) -> list[str]:
    """ Merges the elements column wise to reproduce a single Latex table row.

    Args:
        elements_str (list): table elements in string format

    Returns:
        _type_: Latex table rows
    """
    merged_table_rows = []
    for j in range(len(elements_str)):
        merged_row = ""
        for i in range(len(elements_str[0])): # at this point we assume each row has equal column entries (which they should)
            merged_row += elements_str[j][i]
        merged_table_rows.append(merged_row)
    return merged_table_rows


def ReformatColumns(column_headers : np.array, n : int) -> str:
    """ Reformats columns of the original tables to multi columns and joins them into a single Latex table row

    Args:
        column_headers (np.array): column headers
        n (int) : number of items in each table element

    Returns:
        str: Latex table row
    """
    columns_str = "(legend) " # top left corner is given a legend desribing the contents of each element
    for c in column_headers[1:]: # skip first column which has no entry and is only supossed to be one column
        if n == 1:
            columns_str += f"& {c} "
        else:
            columns_str += "& \\multicolumn{" + str(n) + "}{c|}{" + c + "} "
    columns_str += "\\\\\n"
    return columns_str


def ReformatRowLabels(row_labels : np.array) -> list[str]:
    """ Takes original row labels and replaces them with their fancy counterparts.

    Args:
        row_labels (np.array): original row labels

    Returns:
        list[str]: fancy row labels
    """
    mapping = {
        "" : "",
        "no_selection" : "no selection",
        "calo_size" : "has calorimetry",
        "pi_beam" : "$\\pi^{+}$ beam",
        "pandora_tag" : "pandora tag",
        "dxy" : "$\\delta_{xy}$",
        "dz" : "$\\delta_{z}$",
        "cos_theta" : "$\\cos(\\theta)$",
        "beam_endPos_z" : "beam end z",
        "michel_score" : "michel score",
        "median_dEdX" : "median $dEdX$",
        "track_score" : "track score",
        "em_score" : "em score",
        "beam_separation" : "beam separation",
        "impact_parameter" : "impact parameter",
        "nHits" : "nHits",
        "mass" : "invariant mass",
        "angle" : "opening angle",
        "beam_scraper" : "beam scraper"
    }
    fancy_row_labels = [mapping[i] for i in row_labels]
    return fancy_row_labels


def CreateMergedTable(merged_rows : list, formatted_columns : str, row_labels : np.array) -> list[str]:
    """ Takes the new rows, columns and tables and creates the merged latex table.

    Args:
        merged_rows (list): new rows of tables
        formatted_columns (str): formatted column headers
        row_labels (np.array): formatted row labels

    Returns:
        list[str]: new Latex table
    """
    merged_table = [formatted_columns]
    for label, rows in zip(row_labels[1:], merged_rows):
        merged_table.append(label + " " + rows + " \\\\\n")
    return merged_table


def AddLatexCode(merged_table : list, column_headers : np.array, n : int, add_resize_box : bool = True, table_col_sep : int = 2, hskip : bool = True) -> list[str]:
    """ Adds latex code to the table so it actually compiles.

    Args:
        merged_table (list): Latex table (without latex code)
        column_headers (np.array): column headers
        n (int): number of items per element
        add_resize_box (bool, optional): should resize box be added? Defaults to True.

    Returns:
        list[str]: Latex table with tabular code.
    """
    tab = "\\begin{tabular}{|l|"

    c_fmt = "".join(["c"]*n) + "|"
    for i in range(len(column_headers)):
        tab += c_fmt
    tab += "} \n"

    hline = "\\hline\n"

    final_table = [tab, hline, merged_table[0], hline]
    for m in merged_table[1:]:
        final_table.append(m)

    final_table.append(hline)
    final_table.append("\\end{tabular}\n")
    if add_resize_box:
        final_table.insert(0, "\\resizebox{1.05\linewidth}{!}{\n")
        final_table.append("}\n")

    if table_col_sep:
        final_table.insert(1, "\\renewcommand{\\tabcolsep}{" + str(table_col_sep) + "pt}\n")
    if hskip:
        final_table.insert(1, "\hskip-1cm\n")

    return final_table


def main(args):
    tables = []
    for file in args.files:
        with open(file, "r") as f:
            tables.append(f.readlines())

    split_tables = TablesToArrays(tables)

    merged_elements, row_labels, column_headers = MergeTableArrayElements(split_tables)

    merged_table_rows = MergeElementRows(ReformatElements(merged_elements))

    merged_table = CreateMergedTable(merged_table_rows, ReformatColumns(column_headers, len(args.files)), ReformatRowLabels(row_labels))

    final_table = AddLatexCode(merged_table, column_headers, len(args.files))

    with open(args.out, "w") as f:
        f.writelines(final_table)
        if args.generate_split_tables:
            for i in range(6, len(final_table)-3): # writes the mini tables which only shows the current row and the row before it.
                f.writelines("".join(["-"]*80) + "\n")
                f.writelines(final_table[0:7] + final_table[i:i+2] + final_table[-3:])

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser("reformats tables produced from apps, typically latex produced from pandas. It can also combine tables which have the same rows and column labels")

    parser.add_argument(dest = "files", type = str, nargs = "+", help = "input files, file contains 1 latex table.")
    parser.add_argument("-o", "--output", dest = "out", type = str, default = "table", help = "output file path.")

    parser.add_argument("-s", "--split_tables", dest = "generate_split_tables", action = "store_true", help = "output smaller tables.")

    args = parser.parse_args()

    print(vars(args))
    main(args)