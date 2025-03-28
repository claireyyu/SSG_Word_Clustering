# Data Folder

This folder is used to store the input CSV file for the clustering pipeline.

Expected input file: `accepted_words.csv`

### Format:
The file must have two columns with a header:

```
Word,Original_Frequency
apple,10
banana,5
```

Note: Due to `.gitignore` settings, input files like `accepted_words.csv` are not tracked by Git. You will need to place your own input file here before running the pipeline.
