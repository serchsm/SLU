import pathlib
import pandas as pd

pwd = pathlib.Path.cwd()

train_data = pd.read_csv(pwd.joinpath('fluent_speech_commands_dataset', 'data', 'train_data.csv'))
test_data = pd.read_csv(pwd.joinpath('fluent_speech_commands_dataset', 'data', 'test_data.csv'))
validation_data = pd.read_csv(pwd.joinpath('fluent_speech_commands_dataset', 'data', 'valid_data.csv'))
for file_path, transcript in zip(train_data['path'], train_data['transcription']):
    print(f"{file_path}, {transcript}")