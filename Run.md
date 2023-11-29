
# Guide to Running the Code

## Introduction

This guide provides detailed instructions on how to set up and run the code for the technical challenge. It includes exercises 1.1, 1.2, 2.1, and 2.2.

## Prerequisites

- Python 3.8 or higher installed.
- Access to the terminal or command prompt.
- Download the required dataset from [this link](https://drive.google.com/drive/u/1/folders/1sz96QH7rS3K_07SG_zZms6-iLoxHkdHw) and place it in the `Part2` folder with the name `Software.json`.

## Installation

1. **Clone the Repository**: Clone or download the code repository to your local machine.

### Required Libraries:
- pandas
- beautifulsoup4 (bs4)
- nltk
- matplotlib
- numpy
- transformers
- scikit-learn

### Installation Commands:
```bash
pip3 install pandas
pip3 install beautifulsoup4
pip3 install nltk
pip3 install matplotlib
pip3 install numpy
pip3 install transformers
pip3 install scikit-learn
```


   This will install all the necessary libraries and dependencies required to run the code.

## Running the Code

### Part I

The `Part I` folder contains the following files:

- `Bert_results.txt`
- `comments.json`
- `ideas.json`
- `PreTrainedPortugueseAnalysis.py`
- `innovation.py`
- `innovation12bert.py`
- `cluster_descriptions.txt`
- `innovation_ideas.txt`

#### Steps to Run:

- **Update File Paths**: Change the file_path in all the scripts. Replace `/Users/joaovasco/Desktop/Part I/` with your local directory path.

- **Running Scripts**:
  
  - To run `innovation.py`:
    ```bash
    python3 innovation.py
    ```
  
  - To run `innovation12bert.py`:
    ```bash
    python3 innovation12bert.py
    ```

### Part II

Before running the scripts in Part II, change to the `Part2` directory:

```bash
cd Part2
```

The `Part2` folder contains:

- `21.py`
- `22.py`
- `Software.json`

#### Steps to Run:

- **Place the Dataset**: Ensure `Software.json` is in the `Part2` folder.

- **Running Scripts**:

  - To run `21.py`:
    ```bash
    python3 21.py
    ```
  
  - To run `22.py`:
    ```bash
    python3 22.py
    ```

## Additional Information

- Don't exitate to email me for further questions :)
