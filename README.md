# Project Assignment: Short Video Recommender System (KuaiRec)

## Objective

Develop a recommender system that suggests short videos to users based on user preferences, interaction histories, and video content using the **KuaiRec dataset**. The challenge is to create a personalised and scalable recommendation engine similar to those used in platforms like TikTok or Kuaishou.

## Dataset

We will use the **KuaiRec dataset**, a large-scale, fully-observed dataset collected from the Kuaishou short-video platform.

It contains:

- **User interactions** (views, likes, etc.)
- **Video metadata** (video ID, tags, etc.)
- **Timestamps**

More info: [KuaiRec Paper](https://arxiv.org/abs/2202.10842)

The dataset will be preprocessed and provided in this format:

- `interactions_train.csv`: historical user-item interactions for training.
- `interactions_test.csv`: user-item pairs to score during testing.
- `sample_submission.csv`: a template showing the expected output format.
- `video_metadata.csv`: metadata including tags or content-related features.

### Download the dataset

You can download the dataset via a wget command:

```bash
wget https://nas.chongminggao.top:4430/datasets/KuaiRec.zip --no-check-certificate
unzip KuaiRec.zip
```

### Dataset description

KuaiRec contains millions of user-item interactions as well as side information including the item categories and a social network. Six files are included in the downloaded data:

```bash
KuaiRec
  ├── data
  │   ├── big_matrix.csv          
  │   ├── small_matrix.csv
  │   ├── social_network.csv
  │   ├── user_features.csv
  │   ├── item_daily_features.csv
  │   └── item_categories.csv
  │   └── kuairec_caption_category.csv
```

*Note: To make it work in this repository, place the KuaiRec folder inside a data_final_project/ directory at the root of the project. To do this automatically, feel free to use the FinalProject_Notebook.ipynb.*

## Tasks

1. **Data Preprocessing**
   - Load and inspect the dataset.
   - Handle missing or inconsistent data.
   - Merge metadata for content-based models if necessary.

2. **Feature Engineering**
   - Create meaningful features from interaction and metadata (e.g., content tags, user activity history).
   - Build user-item interaction matrix.
   - Optionally extract time-based or popularity-based features.

3. **Model Development**
   - Choose a recommendation approach:
     - Collaborative filtering (e.g., ALS, Matrix Factorisation)
     - Content-based filtering
     - Sequence-aware models
     - Hybrid approaches
   - Train and validate your model on the training set.

4. **Recommendation Algorithm**
   - Predict which videos are likely to be enjoyed by each user in the test set.
   - Generate a top-N ranked list of recommendations for each user.

5. **Evaluation**
   - Choose suitable metrics (e.g., Precision@K, Recall@K, MAP, NDCG).
   - Evaluate performance and provide interpretations.

**Important note**: This project leaves room for creativity. Different students might take different paths in preprocessing, modelling, and evaluation. What matters is your ability to justify each step with solid reasoning.

## Project architecture

- `EDA/` : Folder containing exploratory data analysis of the provided datasets
- `models/` : Folder containing experiments with different models
- `img/` : Folder containing images used in the notebooks
- `data_final_project/` : Directory where the `KuaiRec` dataset will be stored (automatically downloaded from the main notebook)  
-  `FinalProject_Notebook.ipynb` : Main notebook containing all relevant information

## Evaluation Criteria

- **Functionality**: Does your recommender provide high-quality and relevant video suggestions?
- **Accuracy**: Did you choose meaningful metrics? How well does the model perform according to them?
- **Documentation**: Clear, organised code and explanations of each design choice.
