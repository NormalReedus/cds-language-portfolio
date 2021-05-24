# Assignment 3 - Edge Detection

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
├── assignment_3_venv/
├── create_venv_unix.sh
├── data/
├── edge_detection.py
├── output/
├── README.MD
└── requirements.txt
```

- `./data/` should contain an image from the Jefferson Memorial
    - This image is already included in this repository. Alternatively the image can be downloaded from [this link](https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG) as such:
        - `wget https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG -O ./data/memorial.jpg` - Download image and save in `./data/`
- Run the script `edge_detection.py`:
```bash
python edge_detection.py
```
- `./output/` should now contain three files: `image_with_ROI.jpg`, `image_cropped.jpg`, and `image_letters.jpg`
    - This directory should already have these example output files when cloning the repository
