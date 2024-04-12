import spacy

vocab_file = 'C:\\Users\\pclayadmin\\Desktop\\Projects\\LLM\\training\\input\\word_vocab.txt'
clean_vocab_file = 'C:\\Users\\pclayadmin\\Desktop\\Projects\\LLM\\training\\input\\clean_word_vocab.txt'
vocab = []

with open(vocab_file, 'r', encoding='utf-8', newline='\n') as outfile:
    for word in outfile.readlines():
        new_word = list(word)
        if len(word) == 1:
            continue
        if "/" in new_word or "\\" in new_word or "---" in new_word or "--" in new_word or "+" in new_word or "-" in new_word or '[' in new_word or "]" in new_word:
            continue
        if "\r" in new_word:
            new_word.remove('\r')
        if "\n" in new_word:
            new_word.remove('\n')
        if "\t" in new_word:
            new_word.remove('\t')
        if "," in new_word:
            new_word.remove(',')
        if ")" in new_word:
            new_word.remove(')')
        if "(" in new_word:
            new_word.remove('(')
        if "." in new_word:
            new_word.remove('.')
        if "'" in new_word:
            new_word.remove("'")
        if ":" in new_word:
            new_word.remove(":")
        if "?" in new_word:
            new_word.remove("?")
        if ";" in new_word:
            new_word.remove(";")
        if '"' in new_word:
            new_word.remove('"')
        if '“' in new_word:
            new_word.remove('“')
        if '$' in new_word:
            new_word.remove('$')
        if '’' in new_word:
            new_word.remove('’')
        if '‘' in new_word:
            new_word.remove('‘')
        if '`' in new_word:
            new_word.remove('`')
        if '”' in new_word:
            new_word.remove('”')
        vocab.append(''.join(word))

with open(vocab_file, 'r', encoding='utf-8') as r:
        text = r.read()
        chars = sorted(list(text))

vocab = set(vocab + chars)
        

with open(clean_vocab_file, 'w', encoding='utf-8') as outfile:
    for word in vocab:
        outfile.write(word + '\n')