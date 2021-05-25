# Assignment 4 - Network Analysis

## Prerequisites
You will need to have Bash and Python 3 installed on your device. This script has been tested with Python 3.8.6 on Linux (Ubuntu flavour).
As this has only been tested on Linux, I would recommend running the script on Worker2 or on Windows Subsystem for Linux (Ubuntu) if you are on Windows as the following instructions are written for Linux.

## Installation
- Clone this repository somewhere on your device
- Open a Bash terminal in `/assignment_4/` of the cloned repository or `cd assignment_4` from the root of the repository
- Run the Bash script to generate your virtual environment, generate required directories, and install dependencies:

```bash
./create_venv_unix.sh
```
- If you have any issues with the last step, make sure that you can execute the bash scripts with the following command and try again:

```bash
chmod +x create_venv_unix.sh
```

**NOTE:** Every command from this point will be executed from inside `/assignment_4/`

## Running the script
**NOTE:** Going forwards I will assume you have an alias set for running `python` such that you will not have to type `python3` if that is normally required on your system (usually Mac / Linux). If this is not the case, you will have to substitute `python` with `python3` in the following commands.

- Make sure the newly created virtual environment is activated with:

```bash
source assignment_4_venv/bin/activate
```

Your directory should now have at least these files and directories:

```
.
├── 1_network.py
├── assignment_4_venv/
├── create_venv_unix.sh
├── data/
├── edges/
├── output/
├── README.MD <-- you are here
└── requirements.txt
```

- `./data/` should contain a `.csv` file with news stories 
    - Since the dataset is small, I have included the data used in class when you clone this repo
- Run the script `1_network.py`
    - `-d` or `--data_path` will overwrite the path to the `.csv` file (default is `./data/fake_or_real_news.csv`)
    - `-w` or `--min-weight` sets the minimum number of times an edge will have to occur for it to be drawn in the graph, to not clutter the graph (default is 500)
    - E.g.:
    ```bash
    # default
    python 1_network.py

    # custom data and min-weight
    python 1_network.py -w 1000 -d ./some/path
    ```
- Your terminal should show the progress of the script while running. A `min_weight` under ~1000 can take a long time to draw
- `./output/` should now contain a `.csv` file with centrality measures of the nodes and an image file of a graph with named entities connected by edges if they are mentioned in same news story. The edges are labelled with the number of times they are mentionend together
    - An example graph and centrality measures (run with a minimum weight of 500) are already included when cloning this repo
