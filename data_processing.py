import pathlib
import pandas as pd
import tensorflow as tf
import tensorflow_text

from utilities import utilities
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

train_data = pd.read_csv(pwd.joinpath('fluent_speech_commands_dataset', 'data', 'train_data.csv'))
test_data = pd.read_csv(pwd.joinpath('fluent_speech_commands_dataset', 'data', 'test_data.csv'))
validation_data = pd.read_csv(pwd.joinpath('fluent_speech_commands_dataset', 'data', 'valid_data.csv'))

all_transcripts = pd.concat([train_data, test_data, validation_data], ignore_index=True)
fsc_transcripts = tf.data.Dataset.from_tensor_slices(all_transcripts['transcription'].str.lower().values)

bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[START]", "[END]", "[UNK]"]
bert_vocab_args = dict(vocab_size=1000,
                       reserved_tokens=reserved_tokens,
                       bert_tokenizer_params=bert_tokenizer_params,
                       learn_params={})
fsc_vocabulary = bert_vocab.bert_vocab_from_dataset(fsc_transcripts.batch(100).prefetch(2), **bert_vocab_args)
# print(f"{fsc_vocabulary}")

vocabulary_file_path = pwd.joinpath('resources', 'fsc_vocabulary.txt')
utilities.write_vocabulary(vocabulary_file_path, fsc_vocabulary)

# Example tokenizer
fcs_tokenizer = tensorflow_text.BertTokenizer(str(vocabulary_file_path), **bert_tokenizer_params)
for batch_transcript in fsc_transcripts.batch(3).take(1):
    token_batch = fcs_tokenizer.tokenize(batch_transcript)
    # toke_batch: [batch, num_words, wp] -> [batch, tokens]
    token_batch = token_batch.merge_dims(-2, -1)
    print(f"transcript: {batch_transcript}, tokens: {token_batch}")
    # Tokens to text
    txt_tokens = tf.gather(fsc_vocabulary, token_batch)
    text_commands = tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1)
    print(f"tokens: {token_batch}, text: {text_commands}")
