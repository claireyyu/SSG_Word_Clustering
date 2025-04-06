# Data Folder

This folder is used to store the input CSV files for the clustering pipeline.

## Input Files

### 1. Main Input File: `accepted_words.csv`

#### Format:
The file must have two columns with a header:

```
Word,Original_Frequency
apple,10
banana,5
```

### 2. Custom Word List: `custom_list.csv` (Optional)

#### Format:
The file must have at least one column with a header:

```
word,reason
goat,testing
```

- `word`: The word to filter out (required)
- `reason`: Optional reason for filtering (optional)

Note: Due to `.gitignore` settings, input files like `accepted_words.csv` and `custom_list.csv` are not tracked by Git. You will need to place your own input files here before running the pipeline.
