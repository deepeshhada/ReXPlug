import os
import sys
import argparse
import time
import shutil
import zipfile
import string
import pickle
import requests
import math
from tqdm import tqdm
import pandas as pd


def filter_classes(rating):
	def clamp(n):
		return max(1, min(n, 5))
	return clamp(math.ceil(rating))


def download_file_from_google_drive(id, destination):
	URL = "https://docs.google.com/uc?export=download"
	session = requests.Session()
	response = session.get(URL, params={'id': id}, stream=True)
	token = get_confirm_token(response)
	if token:
		params = {'id': id, 'confirm': token}
		response = session.get(URL, params=params, stream=True)
	save_response_content(response, destination)


def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value
	return None


def save_response_content(response, destination):
	CHUNK_SIZE = 32768
	with open(destination, "wb") as f:
		for chunk in tqdm(response.iter_content(CHUNK_SIZE), position=0, leave=False):
			if chunk:
				f.write(chunk)


def download_files(dataset_name="AmazonDigitalMusic", split_idx="1"):
	with open('./pickled_meta/preprocessed_file_urls.pkl', 'rb') as f:
		split_map = pickle.load(f)

	root_path = os.path.join("./data", dataset_name)
	if not os.path.exists(root_path):
		os.makedirs(root_path)

	for key, value in split_map[dataset_name][split_idx].items():
		print(f'Downloading {key}')
		destination = os.path.join(root_path, key)
		file_id = value
		download_file_from_google_drive(file_id, destination)


def create_discrim_tsv(dataset_name="AmazonDigitalMusic"):
	print('Creating training data for Discriminator.')
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

	df = df.sample(frac=1)
	df.reset_index(drop=True, inplace=True)
	df.to_csv(os.path.join(root_path, 'discrim_train.tsv'), sep='\t', index=False, header=False)


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
		"--split_idx",
		type=str,
		default="1",
		choices=("1", "2", "3", "4", "5"),
		help="Five splits are available for each dataset. Note that argument is string and not int."
	)
	args = parser.parse_args()
	download_files(**(vars(args)))
	create_discrim_tsv(args.dataset_name)
	print('Preprocessed!')
