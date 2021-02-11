import argparse
import os
import pickle
from copy import deepcopy

import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

from collate import CollateTrain, CollateTest
from models.RRCA import *
from utils.rrca_utils import evaluate, train_one_epoch


def get_embeddings(dataset_path):
	with open(os.path.join(dataset_path, 'true_sentence_embeddings.pkl'), 'rb') as f:
		true_embeddings = pickle.load(f)
	return true_embeddings


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


def create_dataset(df, true_embeddings, mode="Test"):
	user_item_ratings = {}
	if mode == "Train":
		for idx, row in df.iterrows():
			user_item_ratings[idx] = [int(row[0]), int(row[1]), true_embeddings[idx], row[3]]
	else:
		for idx, row in df.iterrows():
			user_item_ratings[idx] = [int(row[0]), int(row[1]), row[3]]
	return user_item_ratings


def train_rrca(
	dataset_path="./data",
	model_save_path="./saved_models",
	model="rrca",
	batch_size_rrca=256,
	learning_rate_rrca=0.002,
	num_epochs_rrca=150,
	dataset_name="AmazonDigitalMusic"
):
	with open('./pickled_meta/dataset_meta.pkl', 'rb') as f:
		dataset_meta = pickle.load(f)
	num_users = dataset_meta[dataset_name]['num_users']
	num_items = dataset_meta[dataset_name]['num_items']
	num_factors = 64
	num_layers = 3
	sentence_embed_dim = 512
	embed_dim = num_factors * (2 ** (num_layers - 1))

	model_save_path = os.path.join(model_save_path, dataset_name, model + '.pt')
	dataset_path = os.path.join(dataset_path, dataset_name)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Prepare data_loaders
	train_df = pd.read_csv(os.path.join(dataset_path, 'train_df.csv'))
	val_df = pd.read_csv(os.path.join(dataset_path, 'val_df.csv'))
	test_df = pd.read_csv(os.path.join(dataset_path, 'test_df.csv'))
	print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")

	true_embeddings = get_embeddings(dataset_path)
	user_reviews_dict, item_reviews_dict = create_reviews_lists(train_df, true_embeddings)

	train_set = create_dataset(train_df, true_embeddings, mode="Train")
	val_set = create_dataset(val_df, true_embeddings, mode="Val")
	test_set = create_dataset(test_df, true_embeddings, mode="Test")

	train_loader = DataLoader(
		dataset=train_set,
		batch_size=batch_size_rrca,
		shuffle=True,
		collate_fn=CollateTrain(user_reviews_dict, item_reviews_dict)
	)

	val_loader = DataLoader(
		dataset=val_set,
		batch_size=batch_size_rrca,
		shuffle=False,
		collate_fn=CollateTest(user_reviews_dict, item_reviews_dict)
	)
	test_loader = DataLoader(
		dataset=test_set,
		batch_size=batch_size_rrca,
		shuffle=False,
		collate_fn=CollateTest(user_reviews_dict, item_reviews_dict)
	)

	review_regularizer = ReviewRegularizer(num_factors=num_factors).to(device)
	cross_attention_module = CrossAttention(embed_dim=embed_dim, sentence_embed_dim=sentence_embed_dim).to(device)
	model = RatingPredictor(
		review_regularizer=review_regularizer,
		cross_attention=cross_attention_module,
		embed_dim=embed_dim,
		num_users=num_users,
		num_items=num_items,
		num_factors=num_factors,
		num_layers=num_layers
	).to(device)

	optimizer = optim.Adam(model.parameters(), lr=learning_rate_rrca, weight_decay=0.000001)
	loss_function = nn.MSELoss()
	losses_overall, losses_rating_pred, losses_att, losses_reg = [], [], [], []
	val_mses, val_maes = [], []

	PATIENCE = 15
	patience = PATIENCE
	best_val_mse, best_model = 100, None

	for epoch in range(1, num_epochs_rrca + 1):
		if patience == 0:
			break
		epoch_loss_overall, epoch_loss_rating_pred, epoch_loss_att, epoch_loss_reg, val_mse, val_mae = train_one_epoch(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			loss_function=loss_function,
			optimizer=optimizer,
			epoch=epoch,
			device=device
		)
		if val_mse < best_val_mse:
			print("Saving model...")
			patience = PATIENCE
			best_val_mse = val_mse
			best_model = deepcopy(model)
			torch.save(best_model.state_dict(), model_save_path)
		else:
			patience -= 1
		losses_overall.append(epoch_loss_overall)
		losses_rating_pred.append(epoch_loss_rating_pred)
		losses_att.append(epoch_loss_att)
		losses_reg.append(epoch_loss_reg)
		val_mses.append(val_mse)
		val_maes.append(val_mae)
		print("=" * 80)

	print('RRCA trained. Evaluating on the test set.')
	test_mse, test_mae = evaluate(best_model, test_loader, device)
	print(f"Test MSE: {test_mse} | Test MAE: {test_mae}")
	print("=" * 80)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train ReXPlug.")
	parser.add_argument("--dataset_path", type=str, default="./data", help="Root folder path of preprocessed dataset.")
	parser.add_argument("--model_save_path", type=str, default="./saved_models", help="Root path to save RRCA's model.")
	parser.add_argument("--model", type=str, default="rrca", help="Choose from 'rrca' or 'rr'.")
	parser.add_argument("--batch_size_rrca", type=int, default=256, help="Batch size to train RRCA.")
	parser.add_argument("--learning_rate_rrca", type=int, default=0.002, help="Learning rate for RRCA.")
	parser.add_argument("--num_epochs_rrca", type=int, default=150, help="Number of epochs to train RRCA.")
	parser.add_argument(
		"--dataset_name",
		type=str,
		default="AmazonDigitalMusic",
		choices=("AmazonDigitalMusic", "AmazonVideoGames", "AmazonClothing", "Yelp_1", "Yelp_2", "BeerAdvocate"),
		help="Name of the dataset to use."
	)
	args = parser.parse_args()

	root_path = os.path.join("./data", dataset_name)
	if not os.path.exists(root_path):
		os.makedirs(root_path)
	train_rrca(**(vars(args)))
