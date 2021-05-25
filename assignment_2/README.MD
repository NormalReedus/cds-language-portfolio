# Assignment 2 - String Processing with Python

## Prerequisites
You will need to have Bash and Python 3 installed on your device. This script has been tested with Python 3.8.6 on Linux (Ubuntu flavour).
As this has only been tested on Linux, I would recommend running the script on Worker2 or on Windows Subsystem for Linux (Ubuntu) if you are on Windows as the following instructions are written for Linux.

## Installation
- Clone this repository somewhere on your device
- Open a Bash terminal in `/assignment_2/` of the cloned repository or `cd assignment_2` from the root of the repository
- Run the Bash script to generate your virtual environment, generate required directories, and install dependencies:

```bash
./create_venv_unix.sh
```
- If you have any issues with the last step, make sure that you can execute the bash scripts with the following command and try again:

```bash
chmod +x create_venv_unix.sh
```

**NOTE:** Every command from this point will be executed from inside `/assignment_2/`

## Running the script
**NOTE:** Going forwards I will assume you have an alias set for running `python` such that you will not have to type `python3` if that is normally required on your system (usually Mac / Linux). If this is not the case, you will have to substitute `python` with `python3` in the following commands.

- Make sure the newly created virtual environment is activated with:

```bash
source assignment_2_venv/bin/activate
```

Your directory should now have at least these files and directories:

```
.
├── 1_collocation.py
├── assignment_2_venv/
├── create_venv_unix.sh
├── data/
├── output/
└── README.MD <-- you are here
```


- `./data/` should be filled with arbitrary text files
    - Since the dataset is small, I have already included the corpus of Project Gutenberg novels from class in this repository
- Run the script `1_collocation.py` and supply a target keyword as the first argument
    - Optionally change the directory containing text files with the flag `-d` or `--data_dir`
    - `-w` or `--window_size` will change the number of tokens / words around the keyword to look for collocates in (default is 5)
    - `-s` or `--sample_size` will change the number of text files from `data_dir` that are used. Use this for demo purposes (default is `None`, which will run on all files)
    - E.g.:
    ```bash
    # full run on all files
    python 1_collocation.py sister

    # demo run on custom data with a bigger window size
    python 1_collocation.py sister -d ./some/path -w 10 -s 20
    ```
    - This can take a while when running on all novels (I recommend trying with < 20)
- `./output/` should now contain a `.csv`-file (named after the keyword and window size) with every collocate to the keyword, their raw frequency, and mutual information
    - This directory should already have an example output file when cloning the repository
