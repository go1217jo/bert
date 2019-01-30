# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT language model predict."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import modeling
import tokenization
import numpy as np
import tensorflow as tf
##### changed #####
import random
import math
import copy
#####

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("max_predictions_per_seq", 20,
"In this task, it also refers to maximum number of masked tokens per word.")

flags.DEFINE_string(
		"bert_config_file", None,
		"The config json file corresponding to the pre-trained BERT model. "
		"This specifies the model architecture.")

flags.DEFINE_string(
		"input_file", None,
		"The config json file corresponding to the pre-trained BERT model. "
		"This specifies the model architecture.")

flags.DEFINE_string(
		"output_dir", None,
		"The output directory where the model checkpoints will be written.")

flags.DEFINE_string("vocab_file", None,
										"The vocabulary file that the BERT model was trained on.")

# Other parameters

flags.DEFINE_integer(
	"beam_size", 1,
	"set beam size.")

flags.DEFINE_string(
		"init_checkpoint", None,
		"Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
		"do_lower_case", True,
		"Whether to lower case the input text. Should be True for uncased "
		"models and False for cased models.")

flags.DEFINE_integer(
		"max_seq_length", 128,
		"The maximum total input sequence length after WordPiece tokenization. "
		"Sequences longer than this will be truncated, and sequences shorter "
		"than this will be padded.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
		"tpu_name", None,
		"The Cloud TPU to use for training. This should be either the name "
		"used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
		"url.")

tf.flags.DEFINE_string(
		"tpu_zone", None,
		"[Optional] GCE zone where the Cloud TPU is located in. If not "
		"specified, we will attempt to automatically detect the GCE project from "
		"metadata.")

tf.flags.DEFINE_string(
		"gcp_project", None,
		"[Optional] Project name for the Cloud TPU-enabled project. If not "
		"specified, we will attempt to automatically detect the GCE project from "
		"metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
		"num_tpu_cores", 8,
		"Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
	def __init__(self, unique_id, text):
		self.unique_id = unique_id
		self.text = text


def read_examples(input_file):
	"""Read a list of `InputExample`s from an input file."""
	examples = []
	unique_id = 0
	with tf.gfile.GFile(input_file, "r") as reader:
		while True:
			line = tokenization.convert_to_unicode(reader.readline())
			if not line:
				break
			line = line.strip()
			unique_id += 1
			examples.append(
				InputExample(unique_id, line))
			unique_id += 1
	return examples


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
												 label_ids, scope=None):
	"""Get loss and log probs for the masked LM."""
	input_tensor = gather_indexes(input_tensor, positions)

	with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
		# We apply one more non-linear transformation before the output layer.
		# This matrix is not used after pre-training.
		with tf.variable_scope("transform"):
			input_tensor = tf.layers.dense(
					input_tensor,
					units=bert_config.hidden_size,
					activation=modeling.get_activation(bert_config.hidden_act),
					kernel_initializer=modeling.create_initializer(
							bert_config.initializer_range))
			input_tensor = modeling.layer_norm(input_tensor)

		# The output weights are the same as the input embeddings, but there is
		# an output-only bias for each token.
		output_bias = tf.get_variable(
				"output_bias",
				shape=[bert_config.vocab_size],
				initializer=tf.zeros_initializer())
		logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)
		log_probs = tf.nn.log_softmax(logits, axis=-1)

		"""
		label_ids = tf.reshape(label_ids, [-1])

		one_hot_labels = tf.one_hot(
				label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
		per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])

		loss = tf.reshape(per_example_loss, [-1, tf.shape(positions)[1]])
		"""
		# TODO: dynamic gather from per_example_loss
	# return loss
	return log_probs


def gather_indexes(sequence_tensor, positions):
	"""Gathers the vectors at the specific positions over a minibatch."""
	sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
	batch_size = sequence_shape[0]
	seq_length = sequence_shape[1]
	width = sequence_shape[2]

	flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
	flat_positions = tf.reshape(positions + flat_offsets, [-1])
	flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
	output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
	return output_tensor


def features_to_dict(features, seq_length, max_predictions_per_seq):
	"""Creates an `input_fn` closure to be passed to TPUEstimator."""

	all_input_ids = []
	all_input_mask = []
	all_segment_ids = []
	all_masked_lm_positions = []
	all_masked_lm_ids = []

	for feature in features:
		all_input_ids.append(feature.input_ids)
		all_input_mask.append(feature.input_mask)
		all_segment_ids.append(feature.segment_ids)
		all_masked_lm_positions.append(feature.masked_lm_positions)
		all_masked_lm_ids.append(feature.masked_lm_ids)

	num_examples = len(features)

	feature_dict = {
		"input_ids":
			tf.constant(
				all_input_ids, shape=[num_examples, seq_length],
				dtype=tf.int32),
		"input_mask":
			tf.constant(
				all_input_mask,
				shape=[num_examples, seq_length],
				dtype=tf.int32),
		"segment_ids":
			tf.constant(
				all_segment_ids,
				shape=[num_examples, seq_length],
				dtype=tf.int32),
		"masked_lm_positions":
			tf.constant(
				all_masked_lm_positions,
				shape=[num_examples, max_predictions_per_seq],
				dtype=tf.int32),
		"masked_lm_ids":
			tf.constant(
				all_masked_lm_ids,
				shape=[num_examples, max_predictions_per_seq],
				dtype=tf.int32)
	}

	return feature_dict


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, max_seq_length, tokenizer):
	"""Convert a set of `InputExample`s to a list of `InputFeatures`."""

	all_features = []
	all_tokens = []

	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		features, tokens = convert_single_example(ex_index, example,
																		 max_seq_length, tokenizer)
		all_features.append(features)
		all_tokens.extend(tokens)

	return all_features, all_tokens


def create_masked_lm_prediction(input_ids, mask_position, mask_count=1):
	new_input_ids = list(input_ids)
	masked_lm_labels = []
	masked_lm_positions = list(range(mask_position, mask_position + mask_count))
	for i in masked_lm_positions:
		new_input_ids[i] = 103
		masked_lm_labels.append(input_ids[i])
	return new_input_ids, masked_lm_positions, masked_lm_labels


def process_prediction_input(input_tokens, input_ids, input_mask, segment_ids, max_predictions_per_seq):
	features = []
	new_input_ids = list(input_ids)
	masked_lm_labels = []
	masked_lm_positions = []
	mask_count = 0

	num_prediction = math.ceil(0.15*(len(input_tokens)-2))

	while(mask_count <= num_prediction):
		mask_position = random.randrange(1, len(input_tokens)-1)
		if mask_position in masked_lm_positions:
			continue

		mask_required = 1
		while is_subtoken(input_tokens[mask_position + mask_required]):
			mask_required += 1

		if is_subtoken(input_tokens[mask_position]):
			num_left_mask = 1
			while is_subtoken(input_tokens[mask_position - (num_left_mask)]):
				num_left_mask += 1
			mask_position -= num_left_mask
			mask_required += num_left_mask

		if (mask_count + mask_required) > num_prediction:
			break
		new_mask_positions = list(range(mask_position, mask_position + mask_required))
		for pos in new_mask_positions:
			new_input_ids[pos] = 103
			masked_lm_labels.append(input_ids[pos])
		masked_lm_positions.extend(new_mask_positions)
		mask_count += mask_required

	while len(masked_lm_positions) < max_predictions_per_seq:
		masked_lm_positions.append(0)
		masked_lm_labels.append(0)

	feature = InputFeatures(
		input_ids=new_input_ids,
		input_mask=input_mask,
		segment_ids=segment_ids,
		masked_lm_positions=masked_lm_positions,
		masked_lm_ids=masked_lm_labels)

	return feature


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, segment_ids, input_mask, masked_lm_positions,
							 masked_lm_ids):
		self.input_ids = input_ids,
		self.segment_ids = segment_ids,
		self.input_mask = input_mask,
		self.masked_lm_positions = masked_lm_positions,
		self.masked_lm_ids = masked_lm_ids,


def convert_single_example(ex_index, example, max_seq_length,
													 tokenizer):
	"""Converts a single `InputExample` into a single `InputFeatures`."""
	tokens = tokenizer.tokenize(example.text)
	print("tokens: ", tokens)
	# Account for [CLS] and [SEP] with "- 2"
	if len(tokens) > max_seq_length - 2:
		tokens = tokens[0:(max_seq_length - 2)]

	input_tokens = []
	segment_ids = []
	input_tokens.append("[CLS]")
	segment_ids.append(0)
	for token in tokens:
		input_tokens.append(token)
		segment_ids.append(0)
	input_tokens.append("[SEP]")
	segment_ids.append(0)

	input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

	# The mask has 1 for real tokens and 0 for padding tokens. Only real
	# tokens are attended to.
	input_mask = [1] * len(input_ids)

	# Zero-pad up to the sequence length.
	while len(input_ids) < max_seq_length:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)

	assert len(input_ids) == max_seq_length
	assert len(input_mask) == max_seq_length
	assert len(segment_ids) == max_seq_length

	if ex_index < 5:
		tf.logging.info("*** Example ***")
		tf.logging.info("id: %s" % (example.unique_id))
		tf.logging.info("tokens: %s" % " ".join(
				[tokenization.printable_text(x) for x in input_tokens]))
		tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
		tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
		tf.logging.info("segment_ids: %s" %
										" ".join([str(x) for x in segment_ids]))

	# features = create_sequential_mask(input_tokens, input_ids, input_mask, segment_ids,
	#																	 FLAGS.max_predictions_per_seq)
	##### changed #####
	feature = process_prediction_input(
			input_tokens, input_ids, input_mask, segment_ids, FLAGS.max_predictions_per_seq)

	return feature, input_tokens


def is_subtoken(x):
	return x.startswith("##")


def create_sequential_mask(input_tokens, input_ids, input_mask, segment_ids,
													 max_predictions_per_seq):
	"""Mask each token/word sequentially"""
	features = []
	i = 1
	while i < len(input_tokens) - 1:
		mask_count = 1
		while is_subtoken(input_tokens[i+mask_count]):
			mask_count += 1

		input_ids_new, masked_lm_positions, masked_lm_labels = create_masked_lm_prediction(input_ids, i, mask_count)
		while len(masked_lm_positions) < max_predictions_per_seq:
			masked_lm_positions.append(0)
			masked_lm_labels.append(0)

		feature = InputFeatures(
			input_ids=input_ids_new,
			input_mask=input_mask,
			segment_ids=segment_ids,
			masked_lm_positions=masked_lm_positions,
			masked_lm_ids=masked_lm_labels)
		features.append(feature)
		i += mask_count
	return features


def parse_result(result, all_tokens, features, output_file=None):
	tokenizer = tokenization.FullTokenizer(
			vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
	predict_ids = []
	for item in result:
		predict_ids.append(item['masked_lm_predictions'])

	predict_words = []
	for idx in range(int(len(predict_ids)/20)):
		words_per_example = tokenizer.convert_ids_to_tokens(
				predict_ids[20*idx:20*(idx+1)])
		predict_words.append(words_per_example)

	idx = 0
	for i in range(len(features)):
		origin_sent = []
		while(all_tokens[idx] != "[SEP]"):
			if(all_tokens[idx] == "[CLS]"):
				idx += 1
			else:
				origin_sent.append(all_tokens[idx])
				idx += 1
		idx += 1

		masked_lm_positions = features[i].masked_lm_positions
		num_prediction = np.count_nonzero(masked_lm_positions)
		predict_sent = origin_sent.copy()
		for j in range(num_prediction):
			predict_sent[masked_lm_positions[0][j]-1] = "@" + predict_words[i][j]

		print("origin__sent: ", ' '.join(origin_sent))
		print("predict_sent: ", ' '.join(predict_sent))


def prediction(model, bert_config, masked_index, masked_id, scope=None):
	masked_lm_log_probs = get_masked_lm_output(
			bert_config, model.get_sequence_output(), model.get_embedding_table(),
			tf.constant(masked_index, name="masked_lm_positions"),
			tf.constant(masked_id, name="masked_lm_ids"), scope=scope)
	"""
	tvars = tf.trainable_variables()
	if FLAGS.init_checkpoint:
		(assignment_map, _) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
	tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
	"""
	masked_lm_log_probs = tf.reshape(masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
	output = masked_lm_log_probs

	return output


def beam_search_decoder(data, tokenizer, alpha=1., beam=1):
	print("\ndecoding...")
	sequences = [[list(), 1.0]]
	# walk over each step in sequence
	for m, row in enumerate(data):
		all_candidates = list()
		# expand each current candidate
		Ty = math.pow(float(len(sequences)), alpha)
		for i in range(len(sequences)):
			seq, score = sequences[i]
			sum_value = sum(row)
			#check all likelihood about all words
			for j in range(len(row)):
				candidate = [seq + [j], 1./Ty * score * -math.log(row[j]/sum_value)]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered=sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
		for k in range(10):
			print(tokenizer.convert_ids_to_tokens(ordered[k][0]))
		# select k best
		sequences=ordered[:beam]
		print("\r{}".format(m), end='\n')

	return sequences


def main(_):
	tf.logging.set_verbosity(tf.logging.INFO)

	bert_config=modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

	if FLAGS.max_seq_length > bert_config.max_position_embeddings:
		raise ValueError(
				"Cannot use sequence length %d because the BERT model "
				"was only trained up to sequence length %d" %
				(FLAGS.max_seq_length, bert_config.max_position_embeddings))

	tokenizer=tokenization.FullTokenizer(
		vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
	MASKED_TOKEN="[MASK]"
	MASKED_ID=tokenizer.convert_tokens_to_ids([MASKED_TOKEN])[0]

	lines=[line.rstrip('\n') for line in open(FLAGS.input_file)]
	tokenized_text=[]
	for line in lines:
		if line == '':
			continue
		tokenized_text=tokenized_text + tokenizer.tokenize(line)
		tokenized_text.append('[SEP]')

	tokenized_text.insert(0, '[CLS]')
	token_len=len(tokenized_text)
	origin_text=' '.join(tokenized_text)
	print(tokenized_text)

	indexed_tokens=tokenizer.convert_tokens_to_ids(tokenized_text)
	seps=np.where(np.asarray(indexed_tokens) == 102)[0]
	print(seps)

	segments_ids=np.zeros(token_len, dtype=int)
	if len(seps) > 1:
		segments_ids[seps[0] + 1:]=int(1)
	print(segments_ids)

	ids=tf.placeholder(tf.int32, shape=(1, len(tokenized_text)))
	segments=tf.placeholder(tf.int32, shape=(token_len))

	model=modeling.BertModel(
		config=bert_config,
		is_training=False,
		input_ids=ids,
		token_type_ids=segments,
		use_one_hot_embeddings=True)
			
	skiped_index = []
	
	with tf.Session() as sess:
		prob = None
		print("\ncurrent: {}".format(tokenized_text))
		# sequential masking in each step
		for masked in np.arange(1, token_len, 1):
			masked_index = masked
			# skip [SEP] or number subtoken
			if masked_index in seps or tokenized_text[masked_index].lstrip('##').isdigit():
				skiped_index.append(masked)
				continue
			
			# 원래 토큰 저장
			origin_token=tokenized_text[masked_index]
			# 마스크 되기 전 id
			masked_id=indexed_tokens[masked_index]
			# masking
			tokenized_text[masked_index]=MASKED_TOKEN
			indexed_tokens[masked_index]=MASKED_ID

			# prediction
			outputs=prediction(model, bert_config, [[masked_index]], [[masked_id]], scope=str(masked))
			if masked == 1:
				tvars = tf.trainable_variables()
				if FLAGS.init_checkpoint:
					(assignment_map, _) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
				tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
				sess.run(tf.global_variables_initializer())
			outputs=sess.run(outputs, feed_dict={ ids: [indexed_tokens], segments: segments_ids })

			indexed_tokens[masked_index] = masked_id
			tokenized_text[masked_index] = origin_token
			
			# stack probability distribution
			if type(prob) == np.ndarray:
				prob = np.vstack((prob, outputs))
			else:
				prob = outputs
			print("\r{} / {}".format(masked, token_len), end='')
		
	decoded = beam_search_decoder(prob, tokenizer, alpha=0.7, beam=FLAGS.beam_size)	
	
	if not os.path.exists("results"):
		os.makedirs("results")

	with open("results/cnn_3_3way_alpha_0.7_beam_" + str(FLAGS.beam_size) + ".txt", "wt") as file:
		file.write("[origin]\n" + origin_text)
		for k, seq in enumerate(decoded):
			tokens = tokenizer.convert_ids_to_tokens(seq[0])
			n = 0
			for idx in range(1, len(tokenized_text)-1):
				if idx in skiped_index:
					continue
				tokenized_text[idx] = tokens[n]
				n+=1
			file.write('\n{} candidate:'.format(k))
			for to in tokenized_text:
				file.write('{} '.format(to))

if __name__ == "__main__":
	tf.app.run()
