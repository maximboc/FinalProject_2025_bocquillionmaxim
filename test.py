import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import random
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer

# Data Loading Cell
interactions = pd.read_csv("./data_final_project/KuaiRec/data/big_matrix.csv")
small_interactions = pd.read_csv("./data_final_project/KuaiRec/data/small_matrix.csv")
item_features = pd.read_csv("./data_final_project/KuaiRec/data/item_daily_features.csv")
captions = pd.read_csv("./data_final_project/KuaiRec/data/kuairec_caption_category.csv", lineterminator='\n')

# Data Cleaning Cell
def clean_df(df):
    df = df.dropna()
    df = df.drop_duplicates()  
    return df  

def clean_df_timestamp(df):
    df = clean_df(df)
    df = df[df["timestamp"] >= 0]
    df = df[df["watch_ratio"] <= 200]
    return df

item_features = clean_df(item_features)
item_features = item_features.drop_duplicates(subset='video_id')
captions = clean_df(captions)
captions = captions.drop_duplicates(subset='video_id')
train_df = clean_df_timestamp(interactions)
test_df = clean_df_timestamp(small_interactions)

item_features['upload_dt'] = pd.to_datetime(item_features['upload_dt'])
item_features['date'] = pd.to_datetime(item_features['date'], format='%Y%m%d')

# Clean caption data
captions = captions[['video_id', 'caption', 'first_level_category_name', 'second_level_category_name','third_level_category_name']]
captions.fillna('UNKNOWN', inplace=True)

# Feature Engineering Cell - ALS
to_drop = ['date', 'play_duration', 'video_duration', 'time', 'timestamp']
train_df.drop(columns=to_drop, inplace=True, errors='ignore')
test_df.drop(columns=to_drop, inplace=True, errors='ignore')

train_df['watch_ratio'] = train_df['watch_ratio'].apply(lambda x: min(x, 2.34))
test_df['watch_ratio'] = test_df['watch_ratio'].apply(lambda x: min(x, 2.34))

correlation = item_features[[
       'video_duration', 'video_width',
       'video_height', 'music_id',
       'show_cnt', 'show_user_num', 'play_cnt', 'play_user_num',
       'play_duration', 'complete_play_cnt', 'complete_play_user_num',
       'valid_play_cnt', 'valid_play_user_num', 'long_time_play_cnt',
       'long_time_play_user_num', 'short_time_play_cnt',
       'short_time_play_user_num', 'play_progress', 'comment_stay_duration',
       'like_cnt', 'like_user_num', 'click_like_cnt', 'double_click_cnt',
       'cancel_like_cnt', 'cancel_like_user_num', 'comment_cnt',
       'comment_user_num', 'direct_comment_cnt', 'reply_comment_cnt',
       'delete_comment_cnt', 'delete_comment_user_num', 'comment_like_cnt',
       'comment_like_user_num', 'follow_cnt', 'follow_user_num',
       'cancel_follow_cnt', 'cancel_follow_user_num', 'share_cnt',
       'share_user_num', 'download_cnt', 'download_user_num', 'report_cnt',
       'report_user_num', 'reduce_similar_cnt', 'reduce_similar_user_num',
       'collect_cnt', 'collect_user_num', 'cancel_collect_cnt',
       'cancel_collect_user_num']].corr()

upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
item_features.drop(columns=to_drop, inplace=True, errors='ignore')

item_features['video_age'] = (item_features['date'] - item_features['upload_dt']).dt.days
item_features['is_short_video'] = (item_features['video_duration'].fillna(0) <= 30).astype(int)

to_drop = [
    'date', 'upload_dt', 'video_duration', 'music_id',
    'video_tag_name', 'play_progress', 'video_tag_id',
    'time', 'play_duration'
]
item_features.drop(columns=to_drop, inplace=True, errors='ignore')

train_df = pd.merge(train_df, item_features, on='video_id', how='left')
test_df = pd.merge(test_df, item_features, on='video_id', how='left')

# Feature Engineering Cell - Content-based
def detect_language(text):
    try:
        return detect(text)
    except:
        return "UNKNOWN"
    
captions['language'] = captions['caption'].apply(detect_language)

# Load the BERT tokenizers
tokenizer_cn = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer_kr = BertTokenizer.from_pretrained("beomi/kcbert-base")

# Tokenize the text based on language
def tokenize_by_lang(text, lang):
    text = str(text)
    if lang == 'zh-cn':
        return ' '.join(tokenizer_cn.tokenize(text))
    elif lang == 'ko':
        return ' '.join(tokenizer_kr.tokenize(text))
    else:
        return 'UNKNOWN'

captions['tokenized'] = captions.apply(lambda x: tokenize_by_lang(x['caption'], x['language']), axis=1)

# Engagement Score Cell
def build_engagement_score(df):
    df["engagement_score"] = 0
    
    # Watch Ratio
    if 'watch_ratio' in df.columns:
        df["engagement_score"] += df['watch_ratio'].fillna(0) * 10
    
    # is_short_video
    df["engagement_score"] += df['is_short_video'].fillna(0) * 3
    
    # Video age
    if 'video_age' in df.columns:
        max_age = 365
        normalized_age = np.minimum(df['video_age'].fillna(max_age), max_age) / max_age
        # Newer videos get up to 2 points bonus
        df["engagement_score"] += (1 - normalized_age) * 2
    
    
    if 'video_type' in df.columns:
        df["engagement_score"] += np.where(
            df['video_type'] == 'AD',
            -3,  # penalty for ads
            2    # bonus for regular content
        )
    
    if 'visible_status' in df.columns:
        df["engagement_score"] += np.where(
            df['visible_status'] == 'public',
            2,  
            -1
        )
    
    if 'upload_type' in df.columns:
        upload_type_weights = {
            'ShortImport': 3,     # Short imported videos tend to be high quality
            'StartCamera': 2.5,
            'Knowle': 2,
            'Web': 1.5,
            'LongImport': 1,
            'UNKNOWN': 0,
            'LongCamera': 0.5,
            'PictureSet': 0.5,
            'LongPicture': 0.5,
            'ACurlVideo': 0.5,
            'followShot': 0.5,
            'ShareFromOtherApp': 0.5,
            'SameFrame': 0,
            'PictureCopy': 0,
            'FlashPhoto': 0,
            'PhotoCopy': 0,
            'LocalCollection': 0,
            'LocalInteraction': 0
        }
        df["engagement_score"] += df['upload_type'].map(upload_type_weights).fillna(0)
    
    engagement_columns = {
        'like_cnt': 0.5,
        'comment_cnt': 0.7,
        'share_cnt': 0.8,
        'collect_cnt': 0.6,
        'follow_cnt': 0.9,
        'complete_play_cnt': 0.7,
        'valid_play_cnt': 0.5,
        'reply_comment_cnt': 0.6,
        'comment_like_cnt': 0.4
    }
    
    for col, weight in engagement_columns.items():
        if col in df.columns:
            df["engagement_score"] += np.minimum(np.log1p(df[col].fillna(0)) * weight, 10)

    penalty_columns = {
        'cancel_like_cnt': 0.4,
        'cancel_follow_cnt': 0.5,
        'report_cnt': 0.7
    }
    for col, weight in penalty_columns.items():
        if col in df.columns:
            df["engagement_score"] -= np.minimum(np.log1p(df[col].fillna(0)) * weight, 10)

    if 'video_width' in df.columns:
        df["engagement_score"] += np.where(df['video_width'] >= 720, 0.5, 0)

    if 'video_height' in df.columns:
        df["engagement_score"] += np.where(df['video_height'] >= 1280, 0.5, 0)
    
    # Normalize the score
    min_score = df["engagement_score"].min()
    max_score = df["engagement_score"].max()
    df["engagement_score"] = (df["engagement_score"] - min_score) / (max_score - min_score)
    
    return df

test_df = build_engagement_score(test_df)
train_df = build_engagement_score(train_df)

# ALS Model Cell
user_ids_train = train_df['user_id'].unique()
video_ids_train = train_df['video_id'].unique()

user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids_train)}
video_to_index = {video_id: idx for idx, video_id in enumerate(video_ids_train)}
index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
index_to_video = {idx: video_id for video_id, idx in video_to_index.items()}

train_df['user_index'] = train_df['user_id'].map(user_to_index)
train_df['video_index'] = train_df['video_id'].map(video_to_index)

row = train_df['user_index'].values
col = train_df['video_index'].values
data = train_df['engagement_score'].values

n_users = train_df['user_index'].max() + 1
n_items = train_df['video_index'].max() + 1
    
user_item_matrix = csr_matrix((data, (row, col)), shape=(n_users, n_items))

model = AlternatingLeastSquares(
    factors=100,
    regularization=0.1,
    iterations=15,
    use_gpu=False,
    alpha=40
)

model.fit(user_item_matrix.T)

# Content-based Model Cell
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(captions['tokenized'])
text_sim = cosine_similarity(tfidf_matrix)

n_videos = len(captions)

first_level_sim = (captions['first_level_category_name'].values[:, None] == captions['first_level_category_name'].values).astype(float)
second_level_sim = (captions['second_level_category_name'].values[:, None] == captions['second_level_category_name'].values).astype(float)
third_level_sim = (captions['third_level_category_name'].values[:, None] == captions['third_level_category_name'].values).astype(float)

text_weight = 0.5
first_level_weight = 0.3
second_level_weight = 0.15
third_level_weight = 0.05

combined_sim = (
    text_sim * text_weight +
    first_level_weight * first_level_sim +
    second_level_weight * second_level_sim +
    third_level_weight * third_level_sim
)

combined_sim = np.clip(combined_sim, 0, 1)

indices = pd.Series(captions.index, index=captions['video_id']).drop_duplicates()

# Recommendation Functions Cell
def get_als_recommendations(model, user_item_matrix, user_id, user_to_index, video_to_index, index_to_video, n=10):
    if user_id not in user_to_index:
        return []
    
    user_idx = user_to_index[user_id]
    already_interacted = set(user_item_matrix[user_idx].indices)
    scores = model.user_factors[user_idx].dot(model.item_factors.T)
    
    item_scores = [(item_id, scores[item_id])
                  for item_id in range(len(scores))
                  if item_id not in already_interacted]
    item_scores.sort(key=lambda x: x[1], reverse=True)
    top_items = [index_to_video[item[0]] for item in item_scores[:n]]
    return top_items

def get_content_recommendations(video_id, indices, combined_sim, captions_df, num_recommend=10):
    if video_id not in indices:
        return []
    
    idx = indices[video_id]
    
    if idx >= combined_sim.shape[0]:
        return []
    
    sim_scores = list(enumerate(combined_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar = sim_scores[1:num_recommend+1]
    video_indices = [i[0] for i in top_similar]
    
    valid_indices = [i for i in video_indices if i < len(captions_df)]
    
    if not valid_indices:
        return []
        
    return captions_df['video_id'].iloc[valid_indices].tolist()

def get_popular_recommendations(train_df, num_recommend=10):
    video_counts = train_df['video_id'].value_counts().reset_index()
    video_counts.columns = ['video_id', 'count']
    return video_counts.head(num_recommend)['video_id'].tolist()

popular_recs = get_popular_recommendations(train_df, num_recommend=10)

# Hybrid Recommender Cell
def get_hybrid_recommendations(user_id, video_id, model, user_item_matrix, 
                              user_to_index, video_to_index, index_to_video,
                              indices, combined_sim, captions,
                              popular_recs, n=10, weight_als=0.7):
    """
    Get hybrid recommendations combining collaborative filtering (ALS) and content-based approaches
    with fallback to popular items
    """
    
    als_recs = []
    content_recs = []
    
    # Try to get ALS recommendations for the user
    if user_id in user_to_index:
        als_recs = get_als_recommendations(
            model, user_item_matrix, user_id, 
            user_to_index, video_to_index, index_to_video, n=n
        )
    
    # Try to get content-based recommendations for the video
    if video_id in indices:
        content_recs = get_content_recommendations(
            video_id, indices, combined_sim, captions, num_recommend=n
        )
    
    # If both approaches returned results, combine them with weighted scores
    if als_recs and content_recs:
        # Create a score dictionary for each recommendation type
        als_scores = {vid: (n - i) / n for i, vid in enumerate(als_recs)}
        content_scores = {vid: (n - i) / n for i, vid in enumerate(content_recs)}
        
        # Combine all unique videos
        all_videos = set(als_recs) | set(content_recs)
        
        # Calculate hybrid scores
        hybrid_scores = []
        for vid in all_videos:
            als_score = als_scores.get(vid, 0)
            content_score = content_scores.get(vid, 0)
            hybrid_score = weight_als * als_score + (1 - weight_als) * content_score
            hybrid_scores.append((vid, hybrid_score))
        
        # Sort by hybrid score and get top N
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return [vid for vid, _ in hybrid_scores[:n]]
    
    # If only one approach returned results, use those
    elif als_recs:
        return als_recs
    elif content_recs:
        return content_recs
    
    # If no recommendations could be generated, fall back to popular items
    return popular_recs[:n]

# Evaluation Cell
test_df['user_index'] = test_df['user_id'].map(user_to_index)
test_df['video_index'] = test_df['video_id'].map(video_to_index)

train_user_hist = train_df.groupby("user_id")["video_id"].apply(list).to_dict()
test_user_hist = test_df.groupby("user_id")["video_id"].apply(set).to_dict()

def precision_at_k(recommended_items, relevant_items, k):
    if len(recommended_items) > k:
        recommended_items = recommended_items[:k]
    if not recommended_items:
        return 0.0
    
    hit = len(set(recommended_items) & set(relevant_items))
    return hit / min(k, len(recommended_items))

def recall_at_k(recommended_items, relevant_items, k):
    if len(recommended_items) > k:
        recommended_items = recommended_items[:k]
    if not relevant_items:
        return 0.0
    
    hit = len(set(recommended_items) & set(relevant_items))
    return hit / len(relevant_items)

def ndcg_at_k(recommended_items, relevant_items, k):
    if len(recommended_items) > k:
        recommended_items = recommended_items[:k]
    if not recommended_items or not relevant_items:
        return 0.0
    
    # Create a relevance list where 1 if the item is relevant, 0 otherwise
    relevance = [1 if item in relevant_items else 0 for item in recommended_items]
    
    # Calculate DCG
    dcg = 0
    for i, rel in enumerate(relevance):
        # i+1 because we're using 0-based indexing but rank is 1-based
        dcg += rel / np.log2(i + 2)  # log base 2 of rank+1
    
    # Calculate Ideal DCG (IDCG)
    ideal_relevance = [1] * min(len(relevant_items), k)
    idcg = 0
    for i, rel in enumerate(ideal_relevance):
        idcg += rel / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0

def evaluate_hybrid_recommender(train_user_hist, test_user_hist, model, user_item_matrix, 
                               user_to_index, video_to_index, index_to_video,
                               indices, combined_sim, captions, popular_recs, top_k=10):
    hits = 0
    total = 0
    precision_sum = 0
    recall_sum = 0
    ndcg_sum = 0
    
    for user_id in test_user_hist.keys():
        test_videos = test_user_hist.get(user_id, set())
        train_videos = train_user_hist.get(user_id, [])
        
        if not test_videos:
            continue
        
        seed_video = train_videos[-1] if train_videos else None
        
        if not seed_video:
            continue
        
        # Get hybrid recommendations
        recs = get_hybrid_recommendations(
            user_id, seed_video, model, user_item_matrix,
            user_to_index, video_to_index, index_to_video,
            indices, combined_sim, captions, popular_recs, n=top_k
        )
        
        if not recs:
            continue
        
        if any(video in test_videos for video in recs):
            hits += 1
            
        precision = precision_at_k(recs, test_videos, top_k)
        recall = recall_at_k(recs, test_videos, top_k)
        ndcg = ndcg_at_k(recs, test_videos, top_k)
        
        precision_sum += precision
        recall_sum += recall
        ndcg_sum += ndcg
        
        total += 1
    
    print(f"Total users evaluated: {total}")
    
    metrics = {
        f"HR@{top_k}": hits / total if total > 0 else 0,
        f"Precision@{top_k}": precision_sum / total if total > 0 else 0,
        f"Recall@{top_k}": recall_sum / total if total > 0 else 0,
        f"NDCG@{top_k}": ndcg_sum / total if total > 0 else 0
    }
    
    return metrics

# Run Evaluation
top_k = 10
hybrid_metrics = evaluate_hybrid_recommender(
    train_user_hist, test_user_hist, model, user_item_matrix,
    user_to_index, video_to_index, index_to_video,
    indices, combined_sim, captions, popular_recs, top_k
)

print("Hybrid Model Metrics:")
for metric, value in hybrid_metrics.items():
    print(f"{metric}: {value:.4f}")
