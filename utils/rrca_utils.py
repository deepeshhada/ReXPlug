import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm


def evaluate(model, data_loader, device, mode="Train"):
	model.eval()
	mse_fn = nn.MSELoss()
	mae_fn = nn.L1Loss()

	for data in tqdm(data_loader, position=0, leave=False):
		users, items, reviews, ratings, user_reviews, item_reviews, user_key_mask, item_key_mask = tuple(
			element.to(device) for element in data)
		preds, _, _ = model(users, items, user_reviews, item_reviews, user_key_mask, item_key_mask, mode)
		ratings = ratings.float().view(preds['rating_pred'].size())
		mse = mse_fn(preds['rating_pred'], ratings)
		mae = mae_fn(preds['rating_pred'], ratings)

	return mse.item(), mae.item()


def train_one_epoch(model, train_loader, val_loader, loss_function, optimizer, epoch, device):
	model.train()
	epoch_loss_overall = []
	epoch_loss_rating_pred, epoch_loss_att, epoch_loss_reg = [], [], []
	mae_fn = nn.L1Loss()
	lambda_reg, lambda_att = 5, 5

	timer_start = time.time()
	for data in tqdm(train_loader, position=0, leave=False):
		users, items, reviews, ratings, user_reviews, item_reviews, user_key_mask, item_key_mask = tuple(
			element.to(device) for element in data)
		preds, _, _ = model(users, items, user_reviews, item_reviews, user_key_mask, item_key_mask, mode="Train")
		ratings = ratings.float().view(preds['rating_pred'].size())

		loss1 = loss_function(preds['rating_pred'], ratings)
		loss2 = lambda_att * loss_function(preds['att'], reviews)
		loss3 = lambda_reg * loss_function(preds['reg'], reviews)
		loss = loss1 + loss2 + loss3

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_loss_overall.append(loss.item())
		epoch_loss_rating_pred.append(loss1.item())
		epoch_loss_att.append(loss2.item())
		epoch_loss_reg.append(loss3.item())
	timer_end = time.time()

	epoch_loss_overall = np.mean(epoch_loss_overall)
	epoch_loss_rating_pred = np.mean(epoch_loss_rating_pred)
	epoch_loss_att = np.mean(epoch_loss_att)
	epoch_loss_reg = np.mean(epoch_loss_reg)

	val_mse, val_mae = evaluate(model, val_loader, device)

	print(f'\033[4mEpoch {epoch}\033[0m ({time.strftime("%M:%S", time.gmtime(timer_end - timer_start))}):')
	print(f'\t Overall Train MSE: {epoch_loss_rating_pred:.6f}'.expandtabs(4))
	print(f"\t Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}".expandtabs(4))

	return epoch_loss_overall, epoch_loss_rating_pred, epoch_loss_att, epoch_loss_reg, val_mse, val_mae
