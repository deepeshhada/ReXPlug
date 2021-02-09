import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel

from models.pplm_classification_head import ClassificationHead


def to_var(x, requires_grad=False, volatile=False, device='cuda'):
	if torch.cuda.is_available() and device == 'cuda':
		x = x.cuda()
	elif device != 'cuda':
		x = x.to(device)
	return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
	"""
	Masks everything but the k top entries as -infinity (1e10).
	Used to mask logits such that e^-infinity -> 0 won't contribute to the
	sum of the denominator.
	"""
	if k == 0:
		return logits
	else:
		values = torch.topk(logits, k)[0]
		batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
		if probs:
			return torch.where(logits < batch_mins,
							   torch.ones_like(logits) * 0.0, logits)
		return torch.where(logits < batch_mins,
						   torch.ones_like(logits) * -BIG_CONST,
						   logits)


def perturb_past(
	past,
	model,
	last,
	unpert_past=None,
	unpert_logits=None,
	accumulated_hidden=None,
	grad_norms=None,
	stepsize=0.01,
	classifier=None,
	class_label=None,
	loss_type=0,
	num_iterations=3,
	horizon_length=1,
	window_length=0,
	decay=False,
	gamma=1.5,
	kl_scale=0.01,
	device='cuda',
):
	# Generate inital perturbed past
	grad_accumulator = [
		(np.zeros(p.shape).astype("float32"))
		for p in past
	]

	if accumulated_hidden is None:
		accumulated_hidden = 0

	if decay:
		decay_mask = torch.arange(
			0.,
			1.0 + SMALL_CONST,
			1.0 / (window_length)
		)[1:]
	else:
		decay_mask = 1.0

	_, _, _, curr_length, _ = past[0].shape

	if curr_length > window_length and window_length > 0:
		ones_key_val_shape = (
				tuple(past[0].shape[:-2])
				+ tuple([window_length])
				+ tuple(past[0].shape[-1:])
		)

		zeros_key_val_shape = (
				tuple(past[0].shape[:-2])
				+ tuple([curr_length - window_length])
				+ tuple(past[0].shape[-1:])
		)

		ones_mask = torch.ones(ones_key_val_shape)
		ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
		ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

		window_mask = torch.cat(
			(ones_mask, torch.zeros(zeros_key_val_shape)),
			dim=-2
		).to(device)
	else:
		window_mask = torch.ones_like(past[0]).to(device)

	# accumulate perturbations for num_iterations
	loss_per_iter = []
	new_accumulated_hidden = None
	for i in range(num_iterations):
		curr_perturbation = [
			to_var(torch.from_numpy(p_), requires_grad=True, device=device)
			for p_ in grad_accumulator
		]

		# Compute hidden using perturbed past
		perturbed_past = list(map(add, past, curr_perturbation))
		_, _, _, curr_length, _ = curr_perturbation[0].shape
		all_logits, _, all_hidden = model(last, past=perturbed_past)
		hidden = all_hidden[-1]
		new_accumulated_hidden = accumulated_hidden + torch.sum(
			hidden,
			dim=1
		).detach()
		# TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
		logits = all_logits[:, -1, :]
		probs = F.softmax(logits, dim=-1)

		loss = 0.0
		loss_list = []

		if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
			ce_loss = torch.nn.CrossEntropyLoss()
			# TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
			curr_unpert_past = unpert_past
			curr_probs = torch.unsqueeze(probs, dim=1)
			wte = model.resize_token_embeddings()
			for _ in range(horizon_length):
				inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
				_, curr_unpert_past, curr_all_hidden = model(
					past=curr_unpert_past,
					inputs_embeds=inputs_embeds
				)
				curr_hidden = curr_all_hidden[-1]
				new_accumulated_hidden = new_accumulated_hidden + torch.sum(
					curr_hidden, dim=1)

			prediction = classifier(new_accumulated_hidden /
									(curr_length + 1 + horizon_length))

			label = torch.tensor(prediction.shape[0] * [class_label], device=device, dtype=torch.long)
			discrim_loss = ce_loss(prediction, label)
			loss += discrim_loss
			loss_list.append(discrim_loss)

		kl_loss = 0.0
		if kl_scale > 0.0:
			unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
			unpert_probs = (
					unpert_probs + SMALL_CONST *
					(unpert_probs <= SMALL_CONST).float().to(device).detach()
			)
			correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
				device).detach()
			corrected_probs = probs + correction.detach()
			kl_loss = kl_scale * (
				(corrected_probs * (corrected_probs / unpert_probs).log()).sum()
			)
			loss += kl_loss

		loss_per_iter.append(loss.data.cpu().numpy())
		loss.backward()

		if grad_norms is not None:
			grad_norms = [
				torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
				for index, p_ in enumerate(curr_perturbation)
			]
		else:
			grad_norms = [
				(torch.norm(p_.grad * window_mask) + SMALL_CONST)
				for index, p_ in enumerate(curr_perturbation)
			]

		# normalize gradients
		grad = [
			-stepsize *
			(p_.grad * window_mask / grad_norms[
				index] ** gamma).data.cpu().numpy()
			for index, p_ in enumerate(curr_perturbation)
		]

		# accumulate gradient
		grad_accumulator = list(map(add, grad, grad_accumulator))

		# reset gradients, just to make sure
		for p_ in curr_perturbation:
			p_.grad.data.zero_()

		# removing past from the graph
		new_past = []
		for p_ in past:
			new_past.append(p_.detach())
		past = new_past

	# apply the accumulated perturbations to the past
	grad_accumulator = [
		to_var(torch.from_numpy(p_), requires_grad=True, device=device)
		for p_ in grad_accumulator
	]
	pert_past = list(map(add, past, grad_accumulator))

	return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
	name: Optional[str],
	class_label: Union[str, int],
	device: str
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
	if name is None:
		return None, None

	params = DISCRIMINATOR_MODELS_PARAMS[name]
	classifier = ClassificationHead(
		class_size=params['class_size'],
		embed_size=params['embed_size']
	).to(device)
	if "url" in params:
		resolved_archive_file = cached_path(params["url"])
	elif "path" in params:
		resolved_archive_file = params["path"]
	else:
		raise ValueError("Either url or path have to be specified in the discriminator model parameters")
	classifier.load_state_dict(
		torch.load(resolved_archive_file, map_location=device))
	classifier.eval()

	if isinstance(class_label, str):
		if class_label in params["class_vocab"]:
			label_id = params["class_vocab"][class_label]
		else:
			label_id = params["default_class"]

	elif isinstance(class_label, int):
		if class_label in set(params["class_vocab"].values()):
			label_id = class_label
		else:
			label_id = params["default_class"]
	else:
		label_id = params["default_class"]

	return classifier, label_id
