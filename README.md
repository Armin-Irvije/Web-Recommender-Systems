# Web Recommender Systems (WRS) 2026
## Project - Amazon Reviews, Video Games

Provided below is a concise overview of the dataset and its associated metadata.

To facilitate data loading, we utilize the `Pandas` library. The following code snippet shows how to import a Parquet 
file into a DataFrame:

    import pandas as pd
    
    def parquet_loader(path_file: str = "filename.parquet"):
        print(f"[Parquet] Loading data: `{path_file}` file.")
        return pd.read_parquet(path_file)

---

### Data Description

The Video Games dataset is provided in Parquet format. It has been pre-partitioned into training and test sets, 
following an 80/20 split respectively.

The dataset comprises a total of 37,500 instances (interactions), featuring 1,389 unique users and 932 unique items.

The dataset consists of the following fields:
* `item_id`: A unique identifier for the video game.
* `user_id`: A unique identifier for the user.
* `rating`: The numerical score assigned by the user to the item.

Each record (row) represents a specific User-Item interaction, mapping a given user to an item along with their 
corresponding rating.

---

### MetaData Description

The provided metadata file is also supplied in Parquet format and contains detailed attributes for all items within 
the Video Games dataset. This file allows for a deeper analysis of the items beyond simple identifiers.

This file includes the following fields for every item: `item_id`, `title`, `description`, `features`, `categories`, 
`details`, `rating_number`, and `average_rating`.

---

The provided data are a revised, modified, and processed version of the Amazon Reviews 2023 Dataset 
(Video Games category), which can be found at the following link: https://amazon-reviews-2023.github.io/index.html.
