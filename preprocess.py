import os
import argparse
import math
import time
import sys
import shutil
import zipfile
import string
import pickle
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow_hub as hub


def filter_classes(rating):
	def clamp(n):
		return max(1, min(n, 5))
	return clamp(math.ceil(rating))


def create_discrim_tsv(dataset_name="AmazonDigitalMusic", truncate_after=100000):
	print('Creating training data for Discriminator...', end="")
	root_path = os.path.join("./data", dataset_name)
	train_df = pd.read_csv(os.path.join(root_path, 'train_df.csv'))
	val_df = pd.read_csv(os.path.join(root_path, 'val_df.csv'))
	df = pd.concat([train_df, val_df])
	df = df[['rating', 'review']]
	df.columns = ['class', 'text']

	df['class'] = df['class'].apply(filter_classes)

	drop_quantity = min(list(df['class'].value_counts())) * 3
	for rating in df['class'].unique():
		df = df.drop(df[df['class'] == rating].index[drop_quantity:])
		df.reset_index(drop=True, inplace=True)

	print("\rSampling and saving dataframe...", end="")
	df = df.sample(frac=1)
	df.reset_index(drop=True, inplace=True)
	df = df.truncate(after=truncate_after)
	df.to_csv(os.path.join(root_path, 'discrim_train.tsv'), sep='\t', index=False, header=False)
	print('\rCreated training data for Discriminator!')


def process_text(text):
	return str(text)


def remove_nans(df):
	df.loc[(df.rating == 1) & ((df.review == "") | (df.review.isna())), 'review'] = 'worst'
	df.loc[(df.rating == 2) & ((df.review == "") | (df.review.isna())), 'review'] = 'bad'
	df.loc[(df.rating == 3) & ((df.review == "") | (df.review.isna())), 'review'] = 'average'
	df.loc[(df.rating == 4) & ((df.review == "") | (df.review.isna())), 'review'] = 'good'
	df.loc[(df.rating == 5) & ((df.review == "") | (df.review.isna())), 'review'] = 'best'
	return df


def get_embeddings(reviews, split_save_path):
	print("Getting Universal Sentence Encoder...", end="")
	embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
	reviews = reviews.tolist()
	true_embeddings = []
	print(f"\rEmbedding training set reviews ({len(reviews)}) using Universal Sentence Encoder...")
	for review in tqdm(reviews):
		embeddings = embed([review]).numpy()
		true_embeddings.append(embeddings)
	print("Reviews embedded! Saving embeddings...", end="")
	true_embeddings = np.array(true_embeddings).squeeze()
	with open(split_save_path + 'true_sentence_embeddings.pkl', 'wb') as f:
		pickle.dump(true_embeddings, f, pickle.HIGHEST_PROTOCOL)
	print("\rTraining set review embeddings saved!")
	return true_embeddings


def create_reviews_lists(train_df, true_embeddings, split_save_path):
	user_reviews_dict = {}
	item_reviews_dict = {}
	plain_user_reviews_dict = {}
	plain_item_reviews_dict = {}

	print("Creating review lists (documents)...", end="")

	for idx, row in train_df.iterrows():
		if int(row[0]) not in user_reviews_dict:
			user_reviews_dict[int(row[0])] = []
			plain_user_reviews_dict[int(row[0])] = []
		if int(row[1]) not in item_reviews_dict:
			item_reviews_dict[int(row[1])] = []
			plain_item_reviews_dict[int(row[1])] = []
		user_reviews_dict[int(row[0])].append(true_embeddings[idx])
		item_reviews_dict[int(row[1])].append(true_embeddings[idx])
		plain_user_reviews_dict[int(row[0])].append(row[2])
		plain_item_reviews_dict[int(row[1])].append(row[2])

	with open(split_save_path + 'plain_user_reviews_dict.pkl', 'wb') as f:
		pickle.dump(plain_user_reviews_dict, f, pickle.HIGHEST_PROTOCOL)
	with open(split_save_path + 'plain_item_reviews_dict.pkl', 'wb') as f:
		pickle.dump(plain_user_reviews_dict, f, pickle.HIGHEST_PROTOCOL)

	print("\rCreated review lists!")
	return user_reviews_dict, item_reviews_dict


def generate_splits(data, seed=1234):
	def get_count(d, i):
		ids = set(d[i].tolist())
		return ids

	uid_list, iid_list = get_count(data, 'userId'), get_count(data, 'itemId')
	user_num_all = len(uid_list)
	item_num_all = len(iid_list)

	print("===============Start: Raw Data size======================")
	print(f"total examples: {data.shape[0]}")
	print(f"total number of users: {user_num_all}")
	print(f"total number of items: {item_num_all}")
	print("===============End: Raw Data size========================")
	print(f"-" * 60)

	data_train, data_test = train_test_split(data, test_size=0.2, random_state=seed)
	uids_train, iids_train = get_count(data_train, 'userId'), get_count(data_train, 'itemId')
	user_num = len(uids_train)
	item_num = len(iids_train)

	print("===============Start: no-preprocess: Training Data size======================")
	print("total training examples: {}".format(data_train.shape[0]))
	print("total number of users in training data: {}".format(user_num))
	print("total number of items in training data: {}".format(item_num))
	print("===============End: no-preprocess: Training Data size========================")
	print(f"-" * 60)

	user_frequency_dict = data_train['userId'].value_counts().to_dict()
	item_frequency_dict = data_train['itemId'].value_counts().to_dict()
	uid_misses = []
	iid_misses = []
	if user_num != user_num_all or item_num != item_num_all or 1 in user_frequency_dict.values() or 1 in item_frequency_dict.values():
		for uid in range(user_num_all):
			if uid not in uids_train or user_frequency_dict[uid] == 1:
				uid_misses.append(uid)
		for iid in range(item_num_all):
			if iid not in iids_train or item_frequency_dict[iid] == 1:
				iid_misses.append(iid)

	uid_index = []
	for uid in uid_misses:
		index = data_test.index[data_test['userId'] == uid].tolist()
		uid_index.extend(index)
	data_train = pd.concat([data_train, data_test.loc[uid_index]])

	iid_index = []
	for iid in iid_misses:
		index = data_test.index[data_test['itemId'] == iid].tolist()
		iid_index.extend(index)
	data_train = pd.concat([data_train, data_test.loc[iid_index]])

	all_index = list(set().union(uid_index, iid_index))
	data_test = data_test.drop(all_index)

	user_frequency_dict = data_train['userId'].value_counts().to_dict()
	item_frequency_dict = data_train['itemId'].value_counts().to_dict()

	if 0 in user_frequency_dict.values() or 1 in item_frequency_dict.values():
		print('something seems incorrect!')

	data_test, data_val = train_test_split(data_test, test_size=0.5, random_state=seed)
	uid_list_train, iid_list_train = get_count(data_train, 'userId'), get_count(data_train, 'itemId')
	user_num = len(uid_list_train)
	item_num = len(iid_list_train)

	print("===============Start: processed: Training Data size======================")
	print("total training examples: {}".format(data_train.shape[0]))
	print("total number of users in training data: {}".format(user_num))
	print("total number of items in training data: {}".format(item_num))
	print("===============End: processed: Training Data size========================")
	print(f"-" * 60)
	print(f"Train size: {len(data_train)} | Val size: {len(data_val)} | Test size: {len(data_test)}")
	print(f"-" * 60)
	return data_train, data_val, data_test


def preprocess_rrca(dataset_path="./data/raw_datasets/AmazonDigitalMusic.zip", dataset_name="AmazonDigitalMusic", seed=1234):
	split_save_path = "./data/" + dataset_name + "/"
	if not os.path.exists(split_save_path):
		os.makedirs(split_save_path)

	print("Extracting zipped raw dataset...", end="")
	with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
		zip_ref.extractall(split_save_path)
	print("\rRaw dataset JSON extracted!")

	print("Reading JSON and getting statistics...", end="")
	df = pd.read_json(os.path.join(split_save_path, dataset_name + ".json"), lines=True)
	if dataset_name.startswith('Amazon'):
		df = df[['reviewerID', 'asin', 'reviewText', 'overall']]
	df.columns = ['userId', 'itemId', 'review', 'rating']

	num_users = len(df['userId'].unique())
	num_items = len(df['itemId'].unique())
	density = df.shape[0] / (num_users * num_items)
	print(f"\rTotal interactions: {df.shape[0]} | num_users: {num_users} | num_items: {num_items}")
	print(f"Dataset Density: {density * 100:.4f} %")

	df['userId'] = df['userId'].astype('category').cat.codes
	df['itemId'] = df['itemId'].astype('category').cat.codes
	df['review'] = df['review'].apply(lambda review: process_text(review))
	df = remove_nans(df)

	print("Saving full dataframe...", end="")
	df.to_csv(split_save_path + 'df.csv', index=False)
	df = pd.read_csv(split_save_path + 'df.csv')
	print(f"\rGenerating splits with random seed = {seed}...")
	train_df, val_df, test_df = generate_splits(data=df, seed=seed)

	print("Saving splits as dataframes...", end="")
	df.to_csv(split_save_path + 'df.csv', index=False)
	train_df.to_csv(split_save_path + 'train_df.csv', index=False)
	val_df.to_csv(split_save_path + 'val_df.csv', index=False)
	test_df.to_csv(split_save_path + 'test_df.csv', index=False)
	print("\rSplits saved!")

	train_df = pd.read_csv(split_save_path + 'train_df.csv')
	true_embeddings = get_embeddings(train_df["review"], split_save_path)
	# create review user and item lists
	user_reviews_dict, item_reviews_dict = create_reviews_lists(train_df, true_embeddings, split_save_path)

	# Check if a user/item doesn't have at least 2 reviews in train set
	for k in user_reviews_dict:
		if len(user_reviews_dict[k]) == 1:
			print('User ', k)
	for k in item_reviews_dict:
		if len(item_reviews_dict[k]) == 1:
			print('Item ', k)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Download preprocessed datasets.")

	parser.add_argument(
		"--dataset_name",
		type=str,
		default="AmazonDigitalMusic",
		choices=("AmazonDigitalMusic", "AmazonVideoGames", "AmazonClothing", "Yelp_1", "Yelp_2", "BeerAdvocate"),
		help="Name of the dataset to use."
	)
	parser.add_argument(
		"--dataset_path",
		type=str,
		default="./data/raw_datasets/AmazonDigitalMusic.zip",
		help="Path to the raw dataset."
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=1234,
		help="Random seed state for creating train-val-test split."
	)
	parser.add_argument(
		"--truncate_after",
		type=int,
		default=100000,
		help="Max number of examples for discriminator training."
	)
	args = parser.parse_args()
	timer_start = time.time()
	preprocess_rrca(dataset_path=args.dataset_path, dataset_name=args.dataset_name, seed=args.seed)
	create_discrim_tsv(args.dataset_name, args.truncate_after)
	timer_end = time.time()
	print(f"Dataset preprocessed! Took {timer_end - timer_start:.2f} seconds.")
