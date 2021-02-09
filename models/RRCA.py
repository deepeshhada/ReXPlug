import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
	def __init__(self, sentence_embed_dim):
		super(Attention, self).__init__()
		self.embed_dim = sentence_embed_dim
		self.tanh = nn.Tanh()

	def forward(self, query, keys, values, mask):
		"""
			e = 512, k = num_reviews
			query shape   :   N X query_len X embed_dim   : (nqe)
			keys shape    :   N X key_len X embed_dim     : (nke)
			values shape  :   N X key_len X embed_dim     : (nke)
		"""
		energy = torch.einsum("nqe,nke->nqk", [query, keys]).squeeze()

		if mask is not None:
			energy = energy.masked_fill(mask, float("-1e20")).unsqueeze(1)

		attention = torch.softmax(energy, dim=2)
		output = torch.einsum("nqk,nke->nqe", [attention, values])
		return self.tanh(output), attention


class CrossAttention(nn.Module):
	def __init__(self, embed_dim, sentence_embed_dim):
		super(CrossAttention, self).__init__()
		self.transform = nn.Sequential(
			nn.Linear(embed_dim, sentence_embed_dim),
			nn.Dropout(p=0.5)
		)
		self.attention = Attention(sentence_embed_dim)
		self.mlp = nn.Sequential(
			nn.Linear(sentence_embed_dim, sentence_embed_dim * 2),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(sentence_embed_dim * 2, sentence_embed_dim),
			nn.ReLU()
		)
		self.activation = nn.Sigmoid()

	def forward(self, user_query, item_query, user_context, item_context, user_key_mask, item_key_mask, mode="Train"):
		item_query = self.transform(item_query).unsqueeze(dim=1)
		user_output, user_weights = self.attention(item_query, user_context, user_context, user_key_mask)

		user_query = self.transform(user_query).unsqueeze(dim=1)
		item_output, item_weights = self.attention(user_query, item_context, item_context, item_key_mask)

		if mode == "Test":
			user_weights = torch.argmax(user_weights, dim=-1)
			item_weights = torch.argmax(item_weights, dim=-1)
			user_tensor, item_tensor = user_output, item_output
		else:
			user_weights = F.gumbel_softmax(user_weights, hard=True)
			user_tensor = torch.bmm(user_weights.float(), user_context)
			item_weights = F.gumbel_softmax(item_weights, hard=True)
			item_tensor = torch.bmm(item_weights, item_context)

		predicted = self.activation(user_tensor * item_tensor)
		return predicted, user_weights, item_weights


class ReviewRegularizer(nn.Module):
	def __init__(self, num_factors):
		super(ReviewRegularizer, self).__init__()
		input_size = num_factors * 8
		self.model = nn.Sequential(
			nn.Dropout(p=0.7),
			nn.Linear(input_size, 512),
			nn.Sigmoid()
		)

	def forward(self, interaction):
		return self.model(interaction)


class RatingPredictor(nn.Module):
	def __init__(self, review_regularizer, cross_attention, embed_dim, num_users, num_items, num_factors, num_layers):
		super(RatingPredictor, self).__init__()

		self.review_regularizer = review_regularizer
		self.cross_attention = cross_attention

		self.user_embed = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
		self.item_embed = nn.Embedding(num_embeddings=num_items, embedding_dim=embed_dim)
		self.dropout = nn.Dropout(p=0.5)
		MLP_modules = []
		for i in range(num_layers):
			input_size = num_factors * (2 ** (num_layers - i))
			MLP_modules.append(nn.Dropout(p=0.4))
			MLP_modules.append(nn.Linear(input_size, input_size // 2))
			MLP_modules.append(nn.ReLU())
		self.MLP_forward = nn.Sequential(*MLP_modules)
		self.predict = nn.Linear(num_factors, 1)

	def forward(self, user, item, user_context, item_context, user_key_mask, item_key_mask, mode="Train"):
		preds = {}

		user_embed = self.user_embed(user)
		item_embed = self.item_embed(item)
		interaction = torch.cat((user_embed, item_embed), -1)

		output_MLP = self.MLP_forward(self.dropout(interaction))
		preds['rating_pred'] = self.predict(output_MLP).view(-1)

		masked_predict, user_review_idx, item_review_idx = self.cross_attention(
			user_embed,
			item_embed,
			user_context,
			item_context,
			user_key_mask,
			item_key_mask,
			mode
		)
		preds['att'] = masked_predict.squeeze()

		if mode == "Train":
			regularizer = self.review_regularizer(interaction)
			preds['reg'] = regularizer

		return preds, user_review_idx, item_review_idx
