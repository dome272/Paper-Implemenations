# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import collections

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
        {
            'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
            'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
            'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
            'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
            'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
            'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
            'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
            'bert-base-german-cased': "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
            'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
            'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
            'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
            'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
            'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt",
        }
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    print("text", text)
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class WordpieceTokenizer:
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        if isinstance(vocab, str):
            self.vocab = load_vocab(vocab)
        else:
            self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def __len__(self):
        return len(self.vocab)

    def tokens_to_ids(self, tokens):
        output_ids = []
        if not isinstance(tokens[0], list):
            tokens = [tokens]
        for sentence_tokens in tokens:
            ids = []
            for token in sentence_tokens:
                ids.append(self.vocab.get(token, self.vocab.get(self.unk_token)))
            output_ids.append(ids)
        return output_ids

    def ids_to_tokens(self, ids):
        output_tokens = []
        if not isinstance(ids[0], list):
            ids = [ids]
        for sentence_ids in ids:
            tokens = []
            for id_ in sentence_ids:
                tokens.append(self.reverse_vocab.get(id_, self.unk_token))
            output_tokens.append(tokens)
        return output_tokens

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """
        output_sentences = []
        # if not isinstance(text[0], list):
        #     text = [text]
        print(text)
        for sentence in text:
            output_tokens = []
            for token in whitespace_tokenize(sentence):
                chars = list(token)
                if len(chars) > self.max_input_chars_per_word:
                    output_tokens.append(self.unk_token)
                    continue

                is_bad = False
                start = 0
                sub_tokens = []
                while start < len(chars):
                    end = len(chars)
                    cur_substr = None
                    while start < end:
                        substr = "".join(chars[start:end])
                        if start > 0:
                            substr = "##" + substr
                        if substr in self.vocab:
                            cur_substr = substr
                            break
                        end -= 1
                    if cur_substr is None:
                        is_bad = True
                        break
                    sub_tokens.append(cur_substr)
                    start = end

                if is_bad:
                    output_tokens.append(self.unk_token)
                else:
                    output_tokens.extend(sub_tokens)
            output_sentences.append(output_tokens)
        return output_sentences
