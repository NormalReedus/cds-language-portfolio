import os
import csv
import argparse
from pathlib import Path

import pandas as pd
from collections import Counter
from itertools import combinations
from tqdm import tqdm

import spacy
nlp = spacy.load("en_core_web_sm")

import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)

def main(data_path, min_weight):
    outpath = 'output'

    # Load data as data frame
    input_file = os.path.join(data_path)
    data = pd.read_csv(input_file)

    # Extract all named individuals
    # Extract only mentions of PERSONs in the texts
    real_df = data[data["label"]=="REAL"]["text"]

    print('Extracting PERSONs')
    text_entities = []
    for text in tqdm(real_df):
        tmp_entities = []
        doc = nlp(text)
        # for every named entity
        for entity in doc.ents:
            # if that entity is a person
            if entity.label_ == "PERSON":
                # append to temp list
                tmp_entities.append(entity.text)
        # append temp list to main list
        text_entities.append(tmp_entities)


    # Create edgelist by generating lists with every combination of 2 PERSONs 
    print('Generating edge pairs')
    edgelist = []
    for text in tqdm(text_entities):
        # use itertools.combinations() to create edgelist
        edges = list(combinations(text, 2))
        # for each combination - i.e. each pair of 'nodes'
        for edge in edges:
            # append this to final edgelist
            edgelist.append(tuple(sorted(edge)))

    edgepath = os.path.join('edges', 'edgelist.csv')

    # Write headers to edgefile
    with open(edgepath, 'w', encoding='utf-8') as fh_out:
        fh_out.write('nodeA,nodeB\n')

    # Write data
    with open(edgepath, 'a', encoding='utf-8') as fh_out:
        writer = csv.writer(fh_out)
        writer.writerows(edgelist)

    # Load edgelist csv as a list of tuples
    with open(edgepath, newline='') as fh_in:
        reader = csv.reader(fh_in)
        edgelist = [tuple(row) for row in reader]

    # Remove headers if these are present
    if edgelist[0] == ('nodeA', 'nodeB'):
        edgelist = edgelist[1:]

    # Sum up how many times every unique combination of edges occur
    # and set this number as the weight of the unique edge
    print('Counting edges')
    weighted_edges = []

    for key, value in Counter(edgelist).items():
        nodeA = key[0]
        nodeB = key[1]
        weight = value
        weighted_edges.append((nodeA, nodeB, weight))
    
    # Convert to dataframe for easier manipulation
    edges_df = pd.DataFrame(weighted_edges, columns=["nodeA", "nodeB", "weight"])

    # Remove edges that occur few times
    weight_filtered_df = edges_df[edges_df["weight"] > min_weight]

    # Generate a graph with weighted edges
    print('Drawing graph')
    graph = nx.from_pandas_edgelist(weight_filtered_df, 'nodeA', 'nodeB', edge_attr = 'weight')

    # Let networkx decide where to draw everything
    graph_positions = nx.nx_agraph.graphviz_layout(graph, prog="neato")

    # Draw the graph with labels describing the edges' weight
    nx.draw(graph, graph_positions, with_labels=True, node_size=40, font_size=20, )
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, graph_positions, edge_labels = labels)

    # Save the graph
    graph_outpath = os.path.join(outpath, 'network.png')
    plt.savefig(graph_outpath, dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "create a weighted network graph of news stories")
   
    parser.add_argument("-d", "--data_path", type = Path, default = Path('./data/fake_or_real_news.csv'), help = "the path to the csv file containing the news stories")
    parser.add_argument("-w", "--min_weight", type = int, default = 0, help = "the minimum number of times an edge pair should occur to be included in the network")

    args = parser.parse_args()
    
    main(data_path = args.data_path, min_weight = args.min_weight)