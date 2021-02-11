import argparse
import os
import json
from operator import add
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel

from utils.test_utils import create_cond_df

from models.pplm_classification_head import ClassificationHead

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10
DISCRIMINATOR_MODELS_PARAMS = {}


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
		logits = all_logits[:, -1, :]
		probs = F.softmax(logits, dim=-1)

		loss = 0.0
		loss_list = []

		if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
			ce_loss = torch.nn.CrossEntropyLoss()
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


def run_pplm_example(
		pretrained_model="gpt2-medium",
		discrim="generic",
		dataset_path="./data/AmazonDigitalMusic",
		models_path='./saved_models',
		discrim_weights="./saved_models/AmazonDigitalMusic/generator.pt",
		discrim_meta="./saved_models/AmazonDigitalMusic/generator_meta.json",
		rrca_weights="./saved_models/AmazonDigitalMusic/rrca.pt",
		length=50,
		stepsize=0.02,
		temperature=1.0,
		top_k=10,
		sample=True,
		num_iterations=3,
		grad_length=10000,
		horizon_length=1,
		window_length=0,
		decay=False,
		gamma=1.5,
		gm_scale=0.9,
		kl_scale=0.01,
		seed=0,
		no_cuda=False,
		dataset_name="AmazonDigitalMusic",
		num_reviews=15
):
	# set Random seed
	torch.manual_seed(seed)
	np.random.seed(seed)

	# set the device
	device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
	eos_token = '<|endoftext|>'

	set_generic_model_params(discrim_weights, discrim_meta)
	discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim]["pretrained_model"]

	if pretrained_model != discriminator_pretrained_model:
		pretrained_model = discriminator_pretrained_model

	# load pretrained model
	model = GPT2LMHeadModel.from_pretrained(pretrained_model, output_hidden_states=True).to(device)
	model.eval()

	# load tokenizer
	tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

	# Freeze GPT-2 weights
	for param in model.parameters():
		param.requires_grad = False

	generated_rows = []
	for idx, row in cond_df.iterrows():
		class_label = row.predicted_ratings - 1
		cond_text = row.candidate_reviews
		tokenized_cond_text = tokenizer.encode(
			tokenizer.bos_token + cond_text,
			add_special_tokens=False
		)

		pert_gen_tok_texts = full_text_generation(
			model=model,
			tokenizer=tokenizer,
			context=tokenized_cond_text,
			device=device,
			discrim=discrim,
			class_label=class_label,
			length=length,
			stepsize=stepsize,
			temperature=temperature,
			top_k=top_k,
			sample=sample,
			num_iterations=num_iterations,
			grad_length=grad_length,
			horizon_length=horizon_length,
			window_length=window_length,
			decay=decay,
			gamma=gamma,
			gm_scale=gm_scale,
			kl_scale=kl_scale
		)
		pert_gen_text = tokenizer.decode(pert_gen_tok_texts[0].tolist()[0])
		pert_gen_text = pert_gen_text.splitlines()[0].lstrip(eos_token).split(eos_token, 1)[0].rstrip('\n')
		generated_rows.append((pert_gen_text, row.true_reviews))
	generated_df = pd.DataFrame(data=generated_rows, columns=['explanations', 'true_reviews'])
	generated_df.to_csv(os.path.join(dataset_path, 'generated_df.csv'), index=False)
	print('Generated Explanations!')
	return


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


def full_text_generation(
		model,
		tokenizer,
		context=None,
		device="cuda",
		discrim=None,
		class_label=None,
		length=100,
		stepsize=0.02,
		temperature=1.0,
		top_k=10,
		sample=True,
		num_iterations=3,
		grad_length=10000,
		horizon_length=1,
		window_length=0,
		decay=False,
		gamma=1.5,
		gm_scale=0.9,
		kl_scale=0.01,
		**kwargs
):
	classifier, class_id = get_classifier(
		discrim,
		class_label,
		device
	)

	if classifier is not None:
		loss_type = PPLM_DISCRIM
	else:
		raise Exception("Specify discriminator")

	pert_gen_tok_texts = []

	pert_gen_tok_text = generate_text_pplm(
		model=model,
		tokenizer=tokenizer,
		context=context,
		device=device,
		perturb=True,
		classifier=classifier,
		class_label=class_id,
		loss_type=loss_type,
		length=length,
		stepsize=stepsize,
		temperature=temperature,
		top_k=top_k,
		sample=sample,
		num_iterations=num_iterations,
		grad_length=grad_length,
		horizon_length=horizon_length,
		window_length=window_length,
		decay=decay,
		gamma=gamma,
		gm_scale=gm_scale,
		kl_scale=kl_scale
	)
	pert_gen_tok_texts.append(pert_gen_tok_text)

	if device == 'cuda':
		torch.cuda.empty_cache()

	return pert_gen_tok_texts


def generate_text_pplm(
		model,
		tokenizer,
		context=None,
		past=None,
		device="cuda",
		perturb=True,
		classifier=None,
		class_label=None,
		loss_type=0,
		length=100,
		stepsize=0.02,
		temperature=1.0,
		top_k=10,
		sample=True,
		num_iterations=3,
		grad_length=10000,
		horizon_length=1,
		window_length=0,
		decay=False,
		gamma=1.5,
		gm_scale=0.9,
		kl_scale=0.01
):
	output_so_far = None
	if context:
		context_t = torch.tensor(context, device=device, dtype=torch.long)
		while len(context_t.shape) < 2:
			context_t = context_t.unsqueeze(0)
		output_so_far = context_t

	grad_norms = None
	last = None
	unpert_discrim_loss = 0
	loss_in_time = []

	for i in range(length):
		if past is None and output_so_far is not None:
			last = output_so_far[:, -1:]
			if output_so_far.shape[1] > 1:
				_, past, _ = model(output_so_far[:, :-1])

		unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
		unpert_last_hidden = unpert_all_hidden[-1]

		if i >= grad_length:
			current_stepsize = stepsize * 0
		else:
			current_stepsize = stepsize

		# modify the past if necessary
		if not perturb or num_iterations == 0:
			pert_past = past

		else:
			accumulated_hidden = unpert_last_hidden[:, :-1, :]
			accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

			if past is not None:
				pert_past, _, grad_norms, loss_this_iter = perturb_past(
					past,
					model,
					last,
					unpert_past=unpert_past,
					unpert_logits=unpert_logits,
					accumulated_hidden=accumulated_hidden,
					grad_norms=grad_norms,
					stepsize=current_stepsize,
					classifier=classifier,
					class_label=class_label,
					loss_type=loss_type,
					num_iterations=num_iterations,
					horizon_length=horizon_length,
					window_length=window_length,
					decay=decay,
					gamma=gamma,
					kl_scale=kl_scale,
					device=device
				)
				loss_in_time.append(loss_this_iter)
			else:
				pert_past = past

		pert_logits, past, pert_all_hidden = model(last, past=pert_past)
		pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
		pert_probs = F.softmax(pert_logits, dim=-1)

		if classifier is not None:
			ce_loss = torch.nn.CrossEntropyLoss()
			prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
			label = torch.tensor([class_label], device=device, dtype=torch.long)
			unpert_discrim_loss = ce_loss(prediction, label)
		else:
			unpert_discrim_loss = 0

		# Fuse the modified model and original model
		if perturb:
			unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
			pert_probs = ((pert_probs ** gm_scale) * (
					unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
			pert_probs = top_k_filter(pert_probs, k=top_k,
									  probs=True)  # + SMALL_CONST

			# rescale
			if torch.sum(pert_probs) <= 1:
				pert_probs = pert_probs / torch.sum(pert_probs)

		else:
			pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
			pert_probs = F.softmax(pert_logits, dim=-1)

		# sample or greedy
		if sample:
			last = torch.multinomial(pert_probs, num_samples=1)

		else:
			_, last = torch.topk(pert_probs, k=1, dim=-1)

		# update context/output_so_far appending the new token
		output_so_far = (
			last if output_so_far is None
			else torch.cat((output_so_far, last), dim=1)
		)

	return output_so_far


def set_generic_model_params(discrim_weights, discrim_meta):
	if discrim_weights is None:
		raise ValueError('When using a generic discriminator, '
						 'discrim_weights need to be specified')
	if discrim_meta is None:
		raise ValueError('When using a generic discriminator, '
						 'discrim_meta need to be specified')

	with open(discrim_meta, 'r') as discrim_meta_file:
		meta = json.load(discrim_meta_file)
	meta['path'] = discrim_weights
	DISCRIMINATOR_MODELS_PARAMS['generic'] = meta


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--pretrained_model", "-M", type=str, default="gpt2-medium", help="pretrained model name")
	parser.add_argument("--discrim", "-D", type=str, default="generic", choices="generic", help="Discriminator to use")
	parser.add_argument(
		"--dataset_path",
		type=str,
		default="./data",
		help="Root path of dataset (do not include the dataset name)"
	)
	parser.add_argument(
		'--models_path',
		type=str,
		default="./saved_models",
		help='Root path where models are saved (do not include the dataset name)'
	)
	parser.add_argument("--length", type=int, default=50)
	parser.add_argument("--num_reviews", type=int, default=15)
	parser.add_argument("--stepsize", type=float, default=0.02)
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--top_k", type=int, default=10)
	parser.add_argument("--sample", action="store_true", help="Generate from end-of-text as prefix")
	parser.add_argument("--num_iterations", type=int, default=3)
	parser.add_argument("--grad_length", type=int, default=10000)
	parser.add_argument("--window_length", type=int, default=0, help="Length of past which is being optimized")
	parser.add_argument("--horizon_length", type=int, default=1, help="Length of future to optimize over")
	parser.add_argument("--decay", action="store_true", help="whether to decay or not")
	parser.add_argument("--gamma", type=float, default=1.5)
	parser.add_argument("--gm_scale", type=float, default=0.9)
	parser.add_argument("--kl_scale", type=float, default=0.01)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--no_cuda", action="store_true", help="no cuda")
	parser.add_argument(
		"--dataset_name",
		type=str,
		default="AmazonDigitalMusic",
		choices=("AmazonDigitalMusic", "AmazonVideoGames", "AmazonClothing", "Yelp_1", "Yelp_2", "BeerAdvocate"),
		help="Name of the dataset to use."
	)

	args = parser.parse_args()
	root_model_path = os.path.join(args.models_path, args.dataset_name)
	args.rrca_weights = os.path.join(root_model_path, 'rrca.pt')
	args.discrim_weights = os.path.join(root_model_path, 'generator.pt')
	args.discrim_meta = os.path.join(root_model_path, 'generator_meta.json')
	args.dataset_path = os.path.join(args.dataset_path, args.dataset_name)

	cond_df = create_cond_df(args.dataset_name, args.dataset_path, args.rrca_weights, args.num_reviews)
	run_pplm_example(**vars(args))
