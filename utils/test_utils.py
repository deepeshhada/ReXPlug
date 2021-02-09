import os
import pickle
import argparse
import json
import math
from tqdm import tqdm
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from collate import CollateTest
from models.RRCA import *


def get_embeddings(dataset_path):
	with open(os.path.join(dataset_path, 'true_sentence_embeddings.pkl'), 'rb') as f:
		true_embeddings = pickle.load(f)
	return true_embeddings


def create_dataset(df):
	user_item_ratings = {}
	for idx, row in df.iterrows():
		user_item_ratings[idx] = [int(row[0]), int(row[1]), row[3]]
	return user_item_ratings


def create_reviews_lists(train_df, true_embeddings):
	user_reviews_dict = {}
	item_reviews_dict = {}
	for idx, row in train_df.iterrows():
		if int(row[0]) not in user_reviews_dict:
			user_reviews_dict[int(row[0])] = []
		if int(row[1]) not in item_reviews_dict:
			item_reviews_dict[int(row[1])] = []
		user_reviews_dict[int(row[0])].append(true_embeddings[idx])
		item_reviews_dict[int(row[1])].append(true_embeddings[idx])
	return user_reviews_dict, item_reviews_dict


def get_test_loader(dataset_path, test_df):
	train_df = pd.read_csv(os.path.join(dataset_path, 'train_df.csv'))
	true_embeddings = get_embeddings(dataset_path)
	user_reviews_dict, item_reviews_dict = create_reviews_lists(train_df, true_embeddings)
	test_set = create_dataset(test_df)

	test_loader = DataLoader(
		dataset=test_set,
		batch_size=256,
		shuffle=False,
		collate_fn=CollateTest(user_reviews_dict, item_reviews_dict)
	)
	return test_loader


def clamp(n):
	return max(1, min(n, 5))


def create_cond_df(dataset_path, rrca_weights, num_reviews=10):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	with open(os.path.join(dataset_path, 'plain_item_reviews_dict.pkl'), 'rb') as f:
		plain_item_reviews_dict = pickle.load(f)
	test_df = pd.read_csv(os.path.join(dataset_path, 'test_df.csv'))
	test_loader = get_test_loader(dataset_path, test_df)

	# TODO: get num_users and num_items from pickled file
	num_users = 5561
	num_items = 3568
	num_factors = 64
	num_layers = 3
	sentence_embed_dim = 512
	embed_dim = num_factors * (2 ** (num_layers - 1))

	cnt = 0
	rows = []

	best_review_regularizer = ReviewRegularizer(num_factors=num_factors).to(device)
	best_cross_attention_module = CrossAttention(embed_dim=embed_dim, sentence_embed_dim=sentence_embed_dim).to(device)
	best_model = RatingPredictor(
		review_regularizer=best_review_regularizer,
		cross_attention=best_cross_attention_module,
		embed_dim=embed_dim,
		num_users=num_users,
		num_items=num_items,
		num_factors=num_factors,
		num_layers=num_layers
	).to(device)
	print('Loading model now.')
	best_model.load_state_dict(torch.load(rrca_weights))

	for idx, data in enumerate(tqdm(test_loader)):
		if cnt >= num_reviews:
			break
		users, items, reviews, ratings, user_reviews, item_reviews, user_key_mask, item_key_mask = tuple(
			element.to(device) for element in data
		)
		preds, user_indices, item_indices = best_model(
			users, items, user_reviews, item_reviews, user_key_mask, item_key_mask, "Test"
		)

		items = items.squeeze().detach().cpu().numpy()
		item_indices = item_indices.squeeze().detach().cpu().numpy()
		ratings = ratings.squeeze().detach().cpu().numpy()
		pred = preds['rating_pred'].squeeze().detach().cpu().numpy()
		for pred_idx, rating in enumerate(pred):
			if cnt >= num_reviews:
				break
			try:
				true_rating = clamp(math.ceil(ratings[pred_idx]))
				predicted_rating = clamp(math.ceil(pred[pred_idx]))
				true_review = test_df.review.iloc[[cnt]].item()
				try:
					candidate_review = plain_item_reviews_dict[items[pred_idx]][item_indices[pred_idx]]
				except:
					candidate_review = plain_item_reviews_dict[items[pred_idx]][-1]

				if len(candidate_review) < 20 or len(true_review) < 20:
					continue

				rows.append((true_review, true_rating, candidate_review, predicted_rating))
				cnt += 1
			except:
				continue
		df = pd.DataFrame(data=rows, columns=['true_reviews', 'true_ratings', 'candidate_reviews', 'predicted_ratings'])
		df.to_csv(os.path.join(dataset_path, 'cond_df.csv'), index=False)
	return df
