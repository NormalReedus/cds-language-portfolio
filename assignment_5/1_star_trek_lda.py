# standard library
import os
from pprint import pprint
import json
import argparse
from pathlib import Path

# data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# visualisation
import seaborn as sns
from matplotlib import rcParams
rcParams['figure.figsize'] = 20,10 # figure size in inches

# LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from utils import lda_utils # class util

# warnings
import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

def main(data_path, topic_num):
    outpath = 'output'
    # Unfortunately newlines have been parsed as nothing instead of spaces in the original dataset
    # but the script will work just the same
    with open(data_path) as file:
        content = file.read()
        line_dict = json.loads(content)


    # Takes every line being said by every character of every episode of every Star Trek series
    # and concatenates (joined by spaces) into text docs for every episode
    episodes = {}
    for series_name, series in line_dict.items():
        for episode_name, episode in series.items():
            episode_string = ''

            for character_lines in episode.values():
                lines = ' '.join(character_lines)
            
                # Avoid adding empty lines
                if len(lines) != 0:
                    episode_string += ' ' + lines

            # Add the string containing all lines from the episode to our dict
            episode_key = series_name + '_' + episode_name.split()[1]
            episodes[episode_key] = episode_string

    # Explicitly convert to a list for processing
    episode_lines = list(episodes.values())

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(episode_lines, min_count=10, threshold=100) # higher threshold, fewer phrases.
    trigram = gensim.models.Phrases(bigram[episode_lines], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Tokenize, remove stopwords etc
    # just nouns seem to work the best
    processed_lines = lda_utils.process_words(episode_lines, nlp, bigram_mod, trigram_mod, allowed_postags=["NOUN"])

    # Convert every token to an id
    id2word = corpora.Dictionary(processed_lines)

    # Count frequencies of the tokens (ids) collocation within an episode
    corpus = [id2word.doc2bow(episode_lines) for episode_lines in processed_lines]

    # Define the LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=id2word,num_topics=topic_num, 
                                            random_state=420,
                                            chunksize=10,
                                            passes=10,
                                            iterations=100,
                                            per_word_topics=True, 
                                            minimum_probability=0.0)

    # Compute Perplexity
    metrics = f'Perplexity: {lda_model.log_perplexity(corpus)}' # a measure of how good the model is. the lower, the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                        texts=processed_lines, 
                                        dictionary=id2word, 
                                        coherence='c_v')

    coherence_lda = coherence_model_lda.get_coherence()
    metrics += f'\nCoherence Score: {coherence_lda}'
    print() # newline
    print(metrics)

    # Topic overview
    topic_list = lda_model.print_topics()
    print() # newline
    pprint(topic_list)

    # Save metrics and topic list to file
    with open(os.path.join(outpath, 'metrics_and_topics.txt'), 'w') as file:
        file.write(metrics + '\n\n' + str(topic_list))

    # Generate a plot of topics over episodes
    values = list(lda_model.get_document_topics(corpus))

    split = []
    for entry in values:
        topic_prevelance = []
        for topic in entry:
            topic_prevelance.append(topic[1])
        split.append(topic_prevelance)

    df = pd.DataFrame(map(list,zip(*split)))

    # Smooth the prevalence of topics with a rolling mean
    topic_plot = sns.lineplot(data=df.T.rolling(20).mean())
    # Save plot
    topic_plot.figure.savefig(os.path.join(outpath, 'topics.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "do topic analysis of Star Trek episodes")
   
    parser.add_argument("-d", "--data_path", type = Path, default = Path('./data/all_series_lines.json'), help = "the path to the Star Trek json data file")
    parser.add_argument("-t", "--topic_num", default = 12, type = int, help = "the number of topics to identify in the Star Trek episodes")

    args = parser.parse_args()
    
    main(data_path = args.data_path, topic_num = args.topic_num)