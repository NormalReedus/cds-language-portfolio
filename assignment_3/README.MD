# Assignment 3 - Sentiment Analysis

## Prerequisites
You will need to have Bash and Python 3 installed on your device. This script has been tested with Python 3.8.6 on Linux (Ubuntu flavour).
As this has only been tested on Linux, I would recommend running the script on Worker2 or on Windows Subsystem for Linux (Ubuntu) if you are on Windows as the following instructions are written for Linux.

## Installation
- Clone this repository somewhere on your device
- Open a Bash terminal in `/assignment_3/` of the cloned repository or `cd assignment_3` from the root of the repository
- Run the Bash script to generate your virtual environment, generate required directories, and install dependencies:

```bash
./create_venv_unix.sh
```
- If you have any issues with the last step, make sure that you can execute the bash scripts with the following command and try again:

```bash
chmod +x create_venv_unix.sh
```

**NOTE:** Every command from this point will be executed from inside `/assignment_3/`

## Running the script
**NOTE:** Going forwards I will assume you have an alias set for running `python` such that you will not have to type `python3` if that is normally required on your system (usually Mac / Linux). If this is not the case, you will have to substitute `python` with `python3` in the following commands.

- Make sure the newly created virtual environment is activated with:

```bash
source assignment_3_venv/bin/activate
```

Your directory should now have at least these files and directories:

```
.
├── 1_sentiment.py
├── assignment_3_venv/
├── create_venv_unix.sh
├── data/
├── output/
├── README.MD <-- you are here
└── requirements.txt
```

- `./data/` should contain a `.csv` file with news headlines and their publish date
    - Since the dataset is not too large, I have included the data we used in class when the repo is cloned. 
- Run the script `1_sentiment.py`
    - `-d` or `--data_path` will overwrite the path to the `.csv` file (default is `./data/abcnews-date-text.csv`)
    - `-s` or `--sample_num` will only use a subset of headlines for demo purposes (default is `None`, i.e. all headlines). I recommend running the script on < 100,000 headlines
    - E.g.:
    ```bash
    # all headlines
    python 1_sentiment.py

    # demo with own data
    python 1_sentiment.py -s 100000 -d ./some/path
    ```
- `./output/` should now contain an image file with a graph of the rolling averages of the polarity scores over time
    - An example image file (run on the full, included data) is already included in this repo

## What the plots show
Both of the plots are rolling averages, but there is still great variety in the sentiment, and this would be even more erratic if no smoothing was applied.

### What (if any) are the general trends?
Since the smoothed curves are nearly all in a positive integer value space, we can say that headlines generally convey a positive sentiment, but with a cyclical nature.

### What (if any) inferences might you draw from them?
There is a very negative period in ~mid-2019 that would be very interesting to look closer at. This is probably too early for covid, but what else might have been different for that year?

Similarly one could speculate about what happened in the months where there are extremely high (~2007 & ~2020) polarity scores as well as the low points.
