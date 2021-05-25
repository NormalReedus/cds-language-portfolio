# Assignment 5 - Unsupervised Machine Learning - LDA Topic Modelling

## Prerequisites
You will need to have Bash and Python 3 installed on your device. This script has been tested with Python 3.8.6 on Linux (Ubuntu flavour).
As this has only been tested on Linux, I would recommend running the script on Worker2 or on Windows Subsystem for Linux (Ubuntu) if you are on Windows as the following instructions are written for Linux.

## Installation
- Clone this repository somewhere on your device
- Open a Bash terminal in `/assignment_5/` of the cloned repository or `cd assignment_5` from the root of the repository
- Run the Bash script to generate your virtual environment, generate required directories, and install dependencies:

```bash
./create_venv_unix.sh
```
- If you have any issues with the last step, make sure that you can execute the bash scripts with the following command and try again:

```bash
chmod +x create_venv_unix.sh
```

**NOTE:** Every command from this point will be executed from inside `/assignment_5/`

## Running the script
**NOTE:** Going forwards I will assume you have an alias set for running `python` such that you will not have to type `python3` if that is normally required on your system (usually Mac / Linux). If this is not the case, you will have to substitute `python` with `python3` in the following commands.

- Make sure the newly created virtual environment is activated with:

```bash
source assignment_5_venv/bin/activate
```

Your directory should now have at least these files and directories:

```
.
├── 1_star_trek_lda.py
├── assignment_5_venv/
├── create_venv_unix.sh
├── data/
├── output/
├── README.MD <-- you are here
├── requirements.txt
└── utils/
```

- `./data/` should contain `all_series_lines.json` with every spoken line in every series of Star Trek already included when cloning this repo
    - The data is available on Kaggle [here](https://www.kaggle.com/gjbroughton/start-trek-scripts "Kaggle")
- Run the script `1_star_trek_lda.py`
    - `-d` or `--data_path` will overwrite the path to the `.json` file (default is `./data/all_series_lines.json`)
    - `-t` or `--topic_num` sets the number of topics to identify (default is 12)
    - E.g.:
    ```bash
    # default
    python 1_star_trek_lda.py

    # custom data path and number of topics
    python 1_network.py -t 15 -d ./some/path
    ```
- Your terminal should show the coherence and perplexity metrics of the topics as well as the list of topics identified
- `./output/` should now contain a `.txt` file with the topics and their metrics as well as a graph of the distribution of the topics across episodes of Star Trek
    - Mapping the episode-number (x-axis on graph) back to the series, season, and episode was a bit out of the scope of this assignment, but the episode number should correspond to an episode in a system where every series and season is aggregated into one long list of episodes in order of appearance in the original data
    - An example graph and metrics + topics (12 topics) are already included when cloning this repo


# Topic modeling Star Trek - Assignment Description

We found a dataset on kaggle.com containing all the raw character-lines for each episode of each series of Star Trek (excluding Discovery and Picard). Since the dataset ranges from the series' beginning in the 60's (TOS) until Enterprise which is from 2001, we are interested in finding any significant changes in the topics dealt with over the episodes.
In order to do this, we will train an LDA model to consider each individual episode from each series and model the topics according to that. We can then vizualise topic changes over time, and see if there's a connection between episode topic changes and the overall series.

**Note:**

There are some errors in the dataset where newlines have simply been removed instead of substituted with whitespace, which makes many words concatenated together. This, however, is not detrimental to this demo.

## Outputs
The console will print metrics and the topics from all of the episodes as well as save these.
The script will also save a plot in `/week8/topics.png` that shows the relative prevalence of topics across episodes, although they are hard to track back to the exact episode (see note in 'Running the script').

## Results
From the graph we can see that there is a very clear distinction between the topics of (what looks like from the episode spans) different series (around episode #300), but looking more into which series and episodes these data are from, we would have to re-map the list of episodes back to their names - a project for another day.