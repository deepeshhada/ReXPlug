import numpy as np
import torch


class CollateTest:
	def __init__(self, user_reviews_dict, item_reviews_dict):
		self.user_reviews_dict = user_reviews_dict
		self.item_reviews_dict = item_reviews_dict
		self.pad_vector = np.ones(512)

	def get_key_mask(self, max_length, pad_length):
		current_key_mask = torch.zeros([max_length])
		if pad_length != 0:
			current_key_mask[-pad_length:] = 1
		return current_key_mask

	def stack_and_pad(self, review_list, bsz):
		max_length = max([len(reviews) for reviews in review_list])
		review_tensor = torch.empty([bsz, max_length, 512])
		key_mask = torch.empty([bsz, max_length])

		for idx, reviews in enumerate(review_list):
			pad_length = max_length - len(reviews)
			pad = torch.zeros([pad_length, 512])
			key_mask[idx] = self.get_key_mask(max_length, pad_length)
			reviews = torch.cat([torch.from_numpy(reviews), pad])
			review_tensor[idx] = reviews
		return review_tensor, key_mask.bool()

	def __call__(self, batch):
		bsz = len(batch)
		users = [elements[0] for elements in batch]
		items = [elements[1] for elements in batch]
		targets = [elements[2] for elements in batch]

		user_review_list = [np.array(self.user_reviews_dict[user]) for user in users]
		item_review_list = np.array([np.array(self.item_reviews_dict[item]) for item in items])

		user_reviews, user_key_mask = self.stack_and_pad(user_review_list, bsz)
		item_reviews, item_key_mask = self.stack_and_pad(item_review_list, bsz)

		users = torch.tensor(users, dtype=torch.long)
		items = torch.tensor(items, dtype=torch.long)
		reviews = torch.zeros(1)
		targets = torch.tensor(targets, dtype=torch.float)

		return users, items, reviews, targets, user_reviews, item_reviews, user_key_mask, item_key_mask


class CollateTrain:
	def __init__(self, user_reviews_dict, item_reviews_dict):
		self.user_reviews_dict = user_reviews_dict
		self.item_reviews_dict = item_reviews_dict
		self.pad_vector = np.ones(512)

	def get_key_mask(self, max_length, pad_length):
		current_key_mask = torch.zeros([max_length - 1])
		if pad_length != 0:
			current_key_mask[-pad_length:] = 1
		return current_key_mask

	def stack_and_pad(self, review_list, true_reviews, bsz):
		max_length = max([len(reviews) for reviews in review_list])
		review_tensor = torch.empty([bsz, max_length - 1, 512])
		key_mask = torch.empty([bsz, max_length - 1])
		for idx, reviews in enumerate(review_list):
			pad_length = max_length - len(reviews)
			pad = torch.zeros([pad_length, 512])
			key_mask[idx] = self.get_key_mask(max_length, pad_length)
			mask = true_reviews[idx].numpy()
			for i, review in enumerate(reviews):
				if (mask == review).all():
					reviews = np.delete(reviews, i, axis=0)
					break
			reviews = torch.cat([torch.from_numpy(reviews), pad])
			review_tensor[idx] = reviews
		return review_tensor, key_mask.bool()

	def __call__(self, batch):
		bsz = len(batch)
		users = [elements[0] for elements in batch]
		items = [elements[1] for elements in batch]
		reviews = [torch.from_numpy(elements[2]) for elements in batch]
		targets = [elements[3] for elements in batch]

		user_review_list = [np.array(self.user_reviews_dict[user]) for user in users]
		item_review_list = np.array([np.array(self.item_reviews_dict[item]) for item in items])

		user_reviews, user_key_mask = self.stack_and_pad(user_review_list, reviews, bsz)
		item_reviews, item_key_mask = self.stack_and_pad(item_review_list, reviews, bsz)

		users = torch.tensor(users, dtype=torch.long)
		items = torch.tensor(items, dtype=torch.long)
		reviews = torch.stack(reviews, dim=0)
		targets = torch.tensor(targets, dtype=torch.float)

		return users, items, reviews, targets, user_reviews, item_reviews, user_key_mask, item_key_mask
