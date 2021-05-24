# Topic modeling Star Trek

We found a dataset on kaggle.com containing all the raw character-lines for each episode of each series of Star Trek (excluding Discovery and Picard). Since the dataset ranges from the series' beginning in the 60's (TOS) until Enterprise which is from 2001, we are interested in finding any significant changes in the topics dealt with over the episodes.
In order to do this, we will train an LDA model to consider each individual episode from each series and model the topics according to that. We can then vizualise topic changes over time, and see if there's a connection between episode topic changes and the overall series.

**Note:**

There are some errors in the dataset where newlines have simply been removed instead of substituted with whitespace, which makes many words concatenated together. This, however, is not detrimental to this demo.

## How to run
- Clone this repo and open the directory in bash
- Create the venv and install dependencies:
```bash
bash create_lang_venv.sh # for Mac
bash create_lang_venv_win.sh # for Windows
```
- Manually download the language pack for SpaCy:
```bash
python -m spacy download en_core_web_sm
``` 
- Navigate into `week8/`:
```bash
cd week8
```
- Run the script
```bash
python star_trek_lda.py # for Windows
python3 star_trek_lda.py # for Mac
```

## Outputs
The console will print metrics and the topics from all of the episodes.
The script will also save a plot in `/week8/topics.png` that shows the relative prevalence of topics across episodes, although they are hard to track back to the exact episode

## Results
From the graph we can see that there is a very clear distinction between the topics of (what looks like from the episode spans) different series, but looking more into which series and episodes these data are from, we would have to re-map the list of episodes back to their names - a project for another day.