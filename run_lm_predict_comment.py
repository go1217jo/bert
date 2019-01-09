# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
#####

# tensorflow의 flags 객체를 사용
flags = tf.flags
FLAGS = flags.FLAGS

# flags를 통해 기본 값들을 정함, 커맨드라인을 통해 값을 받을 수 있음
# FLAGS.max_predictions_per_seq = 20 과 같이 정의할 수도 있음
# 
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

## Other parameters

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


# Tokenizer 객체 생성
tokenizer = tokenization.FullTokenizer(
  vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
MASKED_TOKEN = "[MASK]"
MASKED_ID = tokenizer.convert_tokens_to_ids([MASKED_TOKEN])[0]


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
      # text가 utf-8(unicode)로 변환
      line = tokenization.convert_to_unicode(reader.readline())
      if not line:
        break
      # 양쪽 공백 지우기
      line = line.strip()
      unique_id += 1
      # 각 줄에 번호를 붙임
      examples.append(
        InputExample(unique_id, line))
      unique_id += 1
  return examples


def model_fn_builder(bert_config, init_checkpoint, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]


    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # masked_lm_example_loss = get_masked_lm_output(
    #     bert_config, model.get_sequence_output(), model.get_embedding_table(),
    #     masked_lm_positions, masked_lm_ids)
    ##### changed #####
    (masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
        bert_config, model.get_sequence_output(), model.get_embedding_table(),
        masked_lm_positions, masked_lm_ids)
    #####

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.PREDICT:
      ##### changed #####
      masked_lm_log_probs = tf.reshape(masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
      masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
      output = {"masked_lm_log_probs": masked_lm_log_probs,
        "masked_lm_predictions": masked_lm_predictions
      }
      
      # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
      #     mode=mode, predictions=masked_lm_example_loss, scaffold_fn=scaffold_fn)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=output, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn




def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
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

    label_ids = tf.reshape(label_ids, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    loss = tf.reshape(per_example_loss, [-1, tf.shape(positions)[1]])
    # TODO: dynamic gather from per_example_loss
  # return loss
  return per_example_loss, log_probs



def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(features, seq_length, max_predictions_per_seq):
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

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]
    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
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
    })

    # d = d.batch(batch_size=batch_size, drop_remainder=False)
    d = d.batch(batch_size=batch_size)
    return d

  return input_fn



# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, max_seq_length, tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  all_features = []
  all_tokens = []

  # example은 InputExample 객체
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
    new_input_ids[i] = MASKED_ID
    masked_lm_labels.append(input_ids[i])
  return new_input_ids, masked_lm_positions, masked_lm_labels


def process_prediction_input(input_tokens, input_ids, input_mask, segment_ids, max_predictions_per_seq):
  """ 입력, [token, position, segment embedding]"""
  features = []
  new_input_ids = list(input_ids)
  masked_lm_labels = []
  masked_lm_positions = []
  mask_count = 0

  # math.ceil() : 올림 함수
  # 15%만 masking
  num_prediction = math.ceil(0.15*(len(input_tokens)-2))

  while(mask_count <= num_prediction):
    mask_position = random.randrange(1,len(input_tokens)-1)
    # 이미 mask된 위치의 단어면 skip
    if mask_position in masked_lm_positions:
      continue

    mask_required = 1
    # 선택된 단어의 다음 단어가 subtoken인 동안
    while is_subtoken(input_tokens[mask_position + mask_required]):
      mask_required += 1

    # 선택된 단어가 subtoken이면
    if is_subtoken(input_tokens[mask_position]):
      num_left_mask = 1
      while is_subtoken(input_tokens[mask_position - (num_left_mask)]):
        num_left_mask +=1
      mask_position -= num_left_mask
      mask_required += num_left_mask

    if (mask_count + mask_required) > num_prediction:
      break
    new_mask_positions = list(range(mask_position, mask_position + mask_required))
    for pos in new_mask_positions:
      new_input_ids[pos] = MASKED_ID
      masked_lm_labels.append(input_ids[pos])
    masked_lm_positions.extend(new_mask_positions)
    mask_count += mask_required

  while len(masked_lm_positions) < max_predictions_per_seq:
    masked_lm_positions.append(0)
    masked_lm_labels.append(0)

  feature = InputFeatures(
    input_ids = new_input_ids,
    input_mask = input_mask,
    segment_ids = segment_ids,
    masked_lm_positions = masked_lm_positions,
    masked_lm_ids = masked_lm_labels)
  
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
  # subtoken을 고려하여 tokenize
  tokens = tokenizer.tokenize(example.text)
  print("tokens: ", tokens)
  # Account for [CLS] and [SEP] with "- 2"
  # [CLS]와 [SEP]을 추가할 여분을 놔두고 최대 sequence 길이를 넘어가면 자름
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

  # vocab은 key가 단어고 value가 인덱스 (OrderedDict임)
  # 토큰화된 단어 리스트를 사전 기반 인덱스 리스트로 변환
  input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  # list * 정수 하면 해당 리스트를 정수만큼 concat한 것과 같음
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  # 각 리스트를 최대 seq 길이와 맞춤
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  # 길이가 같지 않으면 에러를 냄
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
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

  # features = create_sequential_mask(input_tokens, input_ids, input_mask, segment_ids,
  #                                   FLAGS.max_predictions_per_seq)
  ##### changed #####
  feature = process_prediction_input(input_tokens, input_ids, input_mask, segment_ids, FLAGS.max_predictions_per_seq)
  
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
  tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  predict_ids = []
  for item in result:
    predict_ids.append(item['masked_lm_predictions'])

  predict_words = []
  for idx in range(int(len(predict_ids)/20)):
    words_per_example = tokenizer.convert_ids_to_tokens(predict_ids[20*idx:20*(idx+1)])
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
      predict_sent[masked_lm_positions[0][j]-1] = "<" + predict_words[i][j] + ">"
    
    predict = predict_sent.copy()
    l = 0

    for k, word in enumerate(predict_sent):
      if(word.startswith("##")):
        try:
          if(k >= 1):
            predict[l-1] = predict[l-1] + word[2:]
            predict = predict[:l] + predict[l+1:]
            l-=1
        except Exception as e:
          print(e)
          print('predict_sent')
          print(predict_sent)
          print('length: %d k: %d' % (len(predict_sent), k))
      l+=1

    
    print("origin__sent: ", ' '.join(origin_sent))
    print("predict_sent: ", ' '.join(predict_sent))
    print("predict     : ", ' '.join(predict))
    print()

def main(_):
  # INFO 로그를 볼 수 있게 함, 기본 설정은 WARN
  # DEBUG로 설정하면 모든 타입의 로그를 볼 수 있음
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  # 최대 문장 길이 설정 오류, position embedding은 문자열 길이에 종속
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  # 출력 폴더 생성
  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    # Cluster Resolver for Google Cloud TPUs
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  tf.logging.info('run_config 생성')
  # RunConfig with TPU support
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  tf.logging.info('model_fn 생성')
  # bert_config : json 파일로부터 생성된 bert 모델 config
  # 모델 생성
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  tf.logging.info('estimator 생성')
  # TPU를 지원하는 estimator, CPU, GPU도 지원
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.predict_batch_size)

  # 입력 파일을 읽어와서 각 문장에 번호를 붙임 (반환: InputExample 객체 리스트)
  predict_examples = read_examples(FLAGS.input_file)
  features, all_tokens = convert_examples_to_features(predict_examples,
                                          FLAGS.max_seq_length, tokenizer)

  tf.logging.info("***** Running prediction*****")
  tf.logging.info("  Num examples = %d", len(predict_examples))
  tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

  if FLAGS.use_tpu:
    # Warning: According to tpu_estimator.py Prediction on TPU is an
    # experimental feature and hence not supported here
    raise ValueError("Prediction in TPU not supported")

  # max_predictions_per_seq = sequence당 최대 masking될 수
  predict_input_fn = input_fn_builder(
      features=features,
      seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq)

  result = estimator.predict(input_fn=predict_input_fn)
  output_predict_file = os.path.join(FLAGS.output_dir, "test_results.txt")
  parse_result(result, all_tokens, features, output_predict_file)

if __name__ == "__main__":
  tf.app.run()
