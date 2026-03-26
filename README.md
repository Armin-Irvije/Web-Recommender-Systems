# Web Recommender Systems (WRS) 2026

## Instructions on Armin’s code

### How to run the notebooks

Run **every notebook from top to bottom**, **all cells in order**, starting with **Week 6** and moving forward by week number:

1. `week6.ipynb`
2. `week7.ipynb`
3. `week8.ipynb`
4. `Week9.ipynb`
5. `week10_v2.ipynb`
6. `week11.ipynb`

Do **not** skip weeks: later steps depend on outputs from earlier ones.

### `evals_helper.py` (Week 10 and Week 11)

Keep `**evals_helper.py`** in the project root next to the notebooks. **Week 10** and **Week 11** import it for shared evaluation and text-preprocessing helpers; without it, those notebooks will fail on `import evals_helper`.

### Generated folders and downstream use

When you run the pipeline, the repository will create (or fill) these directories:

- `csv_files/` — preprocessed ratings, matrices, and prediction tables used downstream.
- `**models/`** — trained model pickles (`best_knn_model.pkl`, `best_svd_model.pkl`, `best_content_based_model.pkl`) and cached LLM item-description CSVs where applicable.

These folders are required for downstream notebooks; if they are missing or incomplete, later cells will fail when loading data or models.

### Pre-built Gemma3 and Qwen3 description caches (LLM)

Regenerating LLM descriptions with Ollama is **slow and resource-heavy**. For that reason, **pre-generated** description CSV files are expected to be provided (for example `gemma3_llm_item_descriptions.csv` and `qwen3_llm_item_descriptions.csv`).

**Place those files under `models/`.** Week 11 is configured to read and update caches at:

- `models/qwen3_llm_item_descriptions.csv` (when `model_name == "qwen3:4b"`)
- `models/{model_name}_llm_item_descriptions.csv` (for other model names, e.g. `gemma3`)

Ensure `models/` exists before running Week 11 if you are copying files in manually (the notebook also creates `models/` when saving).

You still need a working **Ollama** setup if you run cells that call the API for new items; with a full cache, those calls are skipped for items already in the CSV.

### External data you must supply

- `**train_video_games.parquet`** and `**test_video_games.parquet**` at the project root (Week 6).
- `**datasets/meta_video_games.parquet**` — item metadata (Week 9, Week 10, Week 11).

---

## Python environment

Install a recent **Python 3.10+** interpreter. For a **dedicated virtual environment** do `python -m venv .venv` then activate it and install packages below. 

If you hit **import errors**, **version conflicts**, or **missing packages** while running a notebook, run the `**%pip install …`** cells at the **top** of that notebook first , or install the matching packages from the table below with `pip` in your venv.


| Notebook | Command                                                                                         |
| -------- | ----------------------------------------------------------------------------------------------- |
| Week 6   | `pip install pyarrow fastparquet matplotlib seaborn`                                            |
| Week 7   | `pip install surprise pandas scikit-learn matplotlib seaborn fastparquet pyarrow numpy==1.26.4` |
| Week 8   | `pip install pandas numpy scipy scikit-learn`                                                   |
| Week 9   | `pip install spacy corenlp sentence-transformers`                                               |
| Week 10  | `pip install pandas numpy scipy scikit-learn`                                                   |
| Week 11  | `pip install ollama`                                                                            |


