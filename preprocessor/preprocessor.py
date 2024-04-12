import os
from tqdm import tqdm
import lzma


def xz_files_in_dir(directory, postfix='.xz'):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(postfix) and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files


def encode(text):
    encoding = []
    for c in text.split(' '):
        item = c
        if "\x00" in c:
            continue
        if "\n" in c:
            item = c.split('\n')[0]
        encoding.append(item)
    return set(encoding)


folder_path = 'C:\\Users\\pclayadmin\\Desktop\\Projects\\LLM\\training\\datasets'
prefix_output_path = 'D:\\Projects\\LLM'
output_file_train = os.path.join(prefix_output_path, 'output_words_train.txt')
output_file_val = os.path.join(prefix_output_path, 'output_words_val.txt')
vocab_file = 'C:\\Users\\pclayadmin\\Desktop\\Projects\\LLM\\training\\input\\word_vocab.txt'


files = xz_files_in_dir(folder_path)
total_files = len(files)

split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]
vocab = set()

# Process the training files
with open(output_file_train, 'w', encoding='utf-8') as outfile:
    for count, filename in enumerate(tqdm(files_train, total=len(files_train))):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, 'rt', encoding='utf-8') as infile:
            text = infile.read()
            outfile.write(text)
            words = encode(text)
            characters = set(text)
            vocab.update(characters)
            vocab.update(words)

# Process the validation files
with open(output_file_val, 'w', encoding='utf-8') as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, 'rt', encoding='utf-8') as infile:
            text = infile.read()
            outfile.write(text)
            words = encode(text)
            characters = set(text)
            vocab.update(characters)
            vocab.update(words)
            
# Write the vocab
with open(vocab_file, 'w', encoding='utf-8') as vfile:
    for char in vocab:
        vfile.write(char + '\n')