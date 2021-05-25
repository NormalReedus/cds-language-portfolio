import argparse
from pathlib import Path
import os
import re
import math

def tokenize(input_string):
    # Split on any non-alphanumeric character
    tokenizer = re.compile(r"\W+")
    
    # Tokenize 
    token_list = tokenizer.split(input_string)

    return token_list

def calc_O11(concordances):
    # Number of times keyword is present with given collocates
    O11 = {} 

    for conc in concordances:
        for collocate in conc:
            if collocate in O11:
                O11[collocate] += 1
            else:
                O11[collocate] = 1

    return O11

def calc_O21_C1(concordances, all_tokens, O11_dict):
    O21 = {}
    C1 = {}

    for conc in concordances:
        for collocate in conc:
            total_count = all_tokens.count(collocate)
            C1[collocate] = total_count
            without_keyword = total_count - O11_dict[collocate]
            O21[collocate] = without_keyword
    
    return (O21, C1)

def calc_kwcount_O12(all_tokens, keyword, O11_dict):
    # Number of times keyword is present without any given collocate
    O12 = {}

    # Number of times the keyword occurs across the corpus
    keyword_count = all_tokens.count(keyword)
    for collocate, count in O11_dict.items():
        O12[collocate] = keyword_count - count

    return (keyword_count, O12)

def calc_exp_freq(C1, R1, N):
    # The expected frequency of a given collocate occurring with the keyword
    exp_freq = {}

    for collocate in C1: # could as well have used O12, O21, or O11 to loop through collocates
        exp_freq[collocate] = (R1 * C1[collocate]) / N

    return exp_freq

def calc_mut_inf(O11, exp_freq):
    # The mutual information for any given collocate to the keyword
    mut_inf = {}

    for collocate in O11: # could as well have used O12, O21, or C1 to loop through collocates
        mut_inf[collocate] = math.log(O11[collocate] / exp_freq[collocate])

    return mut_inf

def write_csv(outfile, O11, mut_inf):
    # Write headers to csv - we do this separately from appending, since this also overwrites the old file
    with open(outfile, 'w', encoding='utf-8') as fh:
        fh.write('collocate,raw_frequency,MI\n')

    with open(outfile, 'a', encoding='utf-8') as fh:
        for collocate in O11:
            fh.write(f'{collocate},{O11[collocate]},{mut_inf[collocate]}\n')

            

def main(data_dir, keyword, window_size, sample_num):
    # Output file is named after the keyword and window size
    outpath = 'output'
    outfile = os.path.join(outpath, f'{keyword}_{window_size}.csv')

    # Let's just work with lowercase for everything
    keyword = keyword.lower()

    # The list of all tokens in the corpus
    # Note: concordance lines based off this can actually cross between different novels
    tokens = [] 

    # Note: data_dir is already converted to a Path in ArgumentParser
    for i, novel_path in enumerate(data_dir.glob('*.txt')):
        # Only use a subset of files if running as demo
        if i == sample_num:
            break

        with open(novel_path, 'r', encoding='utf-8') as fh:
            content = fh.read()
            # Splits the whole novel-content string into tokens on non-word characters
            tokens += tokenize(content)

    # Converts all tokens to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Number of total tokens in corpus
    N = len(tokens)

    # Locations of keyword in the token list
    keyword_indices = [i for i, token in enumerate(tokens) if token == keyword.lower()]

    # A list of tokens slices +- window_size around every keyword
    concordances = [tokens[max(0, i - window_size) : i + window_size + 1] for i in keyword_indices]

    # Filters out the keyword that we are checking against, but not OTHER occurences of the keyword
    for conc in concordances:
        # Removes the window_size'th element (exact middle) of concordance line, which is the keyword
        conc.pop(window_size)

    # Number of times keyword is present with given collocates
    O11 = calc_O11(concordances)

    # O21: Number of times given collocate occurs without keyword
    # C1: Total number of times a given collocate occurs
    O21, C1 = calc_O21_C1(concordances, tokens, O11)
    
    # R1 / keyword_count: Number of times the keyword occurs across the corpus
    # O12: Number of times keyword is present without any given collocate
    R1, O12 = calc_kwcount_O12(tokens, keyword, O11)

    # The expected frequency of a given collocate occurring with the keyword
    exp_freq = calc_exp_freq(C1, R1, N)

    # The mutual information for any given collocate to the keyword
    mut_inf = calc_mut_inf(O11, exp_freq)

    # Write 3 cols: collocate, raw_frequency, MI
    write_csv(outfile, O11, mut_inf)
    
    print('Data written to: ' + outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate collocates for a specific keyword')
    parser.add_argument('keyword', help='the keyword to look for')
    parser.add_argument('-w', '--window_size', type=int, default=5, help='the number of words on both sides of the keyword to look for collocates in')
    parser.add_argument('-s', '--sample_num', type=int, help='whether to only use a subset of files and how many to use')
    parser.add_argument('-d', '--data_dir', type=Path, default = Path('./data/'), help='the directory containing all of your text files to analyze')
    args = parser.parse_args()	

    main(keyword = args.keyword, window_size = args.window_size, sample_num = args.sample_num, data_dir = args.data_dir)