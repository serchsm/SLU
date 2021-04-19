import logging
import pathlib
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

train_data = pd.read_csv(pwd.joinpath('fluent_speech_commands_dataset', 'data', 'train_data.csv'))
test_data = pd.read_csv(pwd.joinpath('fluent_speech_commands_dataset', 'data', 'test_data.csv'))
validation_data = pd.read_csv(pwd.joinpath('fluent_speech_commands_dataset', 'data', 'valid_data.csv'))

all_transcripts = pd.concat([train_data, test_data, validation_data], ignore_index=True)

for transcript in all_transcripts['transcription']:
    print(f"{transcript}")

fsc_transcripts = tf.data.Dataset.from_tensor_slices(all_transcripts['transcription'].str.lower().values)

# for transcripts in fsc_transcripts.take(5):
#     print(f"{transcripts}")
# for file_path, transcript in zip(train_data['path'], train_data['transcription']):
#     print(f"{file_path}, {transcript}")

