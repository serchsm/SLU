import pathlib
import time
import pandas as pd
import tensorflow as tf

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
bert_vocab_args = dict(vocab_size=100,
                       reserved_tokens=reserved_tokens,
                       bert_tokenizer_params=bert_tokenizer_params,
                       learn_params={})
fsc_vocabulary = bert_vocab.bert_vocab_from_dataset(fsc_transcripts.batch(100).prefetch(2), **bert_vocab_args)
print(f"{fsc_vocabulary}")

# for transcripts in fsc_transcripts.take(5):
#     print(f"{transcripts}")
# for file_path, transcript in zip(train_data['path'], train_data['transcription']):
#     print(f"{file_path}, {transcript}")

