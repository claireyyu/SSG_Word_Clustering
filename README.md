
# Word Difficulty Clustering

This repository provides a word difficulty clustering pipeline using real-world frequency and spelling complexity features. The process is automated and configurable through a YAML file.

## Directory Structure

```
SSG_Word_Clustering/
├── clustering/                  # Core modules (preprocessing, feature engineering, clustering)
├── data/                        # Input data folder
│   └── accepted_words.csv       # Placeholder for your input word list
├── output/                      # Output folder for results
├── main.py                      # Entry point
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
└── README.md                    # Project instructions
```

---

## 1. Clone the Repository

```bash
git clone https://github.com/claireyyu/SSG_Word_Clustering.git
cd SSG_Word_Clustering
```

---

## 2. Set Up Virtual Environment

It is recommended to use a virtual environment to manage dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Prepare Your Input File

Place your word list in the `data/` directory as `accepted_words.csv`.

### Expected Format:

```csv
Word,Original_Frequency
apple,12.4
banana,5.2
...
```

The file must contain these two columns. The header row is required.

---

## 4. Run the Pipeline

Run the following command to execute the full clustering pipeline:

```bash
python main.py --config config.yaml
```

On Mac, run:
```bash
python3 main.py --config config.yaml
```

---

## 5. Output

After execution, results will be saved in the `output/` directory:

- `clusters_by_label/cluster0_words.csv` – clustered word lists by difficulty
- `final_cleaned_clusters.csv` – full list of cleaned clustered words
- `summary.json` – basic statistics and settings used

---

## 6. Cleaning Up

To deactivate the virtual environment:

```bash
deactivate
```

---

## Notes

- The pipeline automatically downloads required NLTK corpora the first time you run it.
- Non-English or malformed words will be removed during final filtering.
