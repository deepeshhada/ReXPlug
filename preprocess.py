import os
import sys
import argparse
import time
import shutil
import zipfile
import string
import pickle
import requests
from tqdm import tqdm


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
		for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)


def download_files(dataset_name="AmazonDigitalMusic", split_idx="1"):
	# TODO: load pickled split map
	# TODO: Get file_id and destination path from pickled dict
	file_id = '1fiiqvpXgy21qBzvU0zTN8mxTHgFIZYoM'
	destination = 'dataset.zip'
	download_file_from_google_drive(file_id, destination)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Download preprocessed datasets.")
	parser.add_argument(
		"--dataset_name",
		type=str,
		default="AmazonDigitalMusic",
		help="Name of the dataset to use. Select one from AmazonDigitalMusic, AmazonVideoGames, AmazonClothing, Yelp_1, Yelp_2, BeerAdvocate."
	)
	parser.add_argument("--split_idx", type=str, default="1", help="Five splits are available for each dataset.")
	args = parser.parse_args()
	download_files(**(vars(args)))
