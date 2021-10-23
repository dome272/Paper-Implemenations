import random
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from WordPiece import WordpieceTokenizer
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.DEBUG, datefmt="%I:%M:%S")
torch.autograd.set_detect_anomaly(True)

SPECIAL_TOKENS = {
    "CLS": 0,
    "PAD": 1,
    "SEP": 2,
    "MASK": 3,
    "UNK": 4
}


def mlm_accuracy(mlm_pred, mlm_actual):
    """
    BERT Masked Language Model for measuring the accuracy on predicting masked tokens in the input sequence.
    :param mlm_pred: array of predicted sentence tokens
    :param mlm_actual: array of actual sentence tokens
    :return: accuracy score
    """
    relevant_idx = np.where(mlm_actual != SPECIAL_TOKENS["PAD"])
    mlm_pred = mlm_pred[relevant_idx]
    mlm_actual = mlm_actual[relevant_idx]

    return np.equal(mlm_pred, mlm_actual).type(torch.FloatTensor).mean()


def nsp_accuracy(nsp_pred, nsp_actual):
    """
    BERT Next Sentence Prediction for measuring the accuracy on correctly predicting if the following sentence actually
    is the next sentence.
    :param nsp_pred: predicted binary labels
    :param nsp_actual: actual binary labels
    :return: accuracy score
    """
    return np.equal(nsp_pred, nsp_actual).type(torch.FloatTensor).mean()


def padding_batch(batch):
    """
    Padding all elements in the batch to the maximum sequence length in a batch.
    :param batch: sentence (named as sequence here), segmentation for distinguishing between the sentences,
    actual sentence, label for NSP
    :return: padded batch of sentence, segment, target sentence, NSP label
    """
    targets = [target for _, (target, is_next) in batch]
    longest_target = max(targets, key=lambda t: len(t))
    max_len = len(longest_target)

    padded_sequences = []
    padded_segments = []
    padded_targets = []
    is_nexts = []

    for (sequence, segment), (target, is_next) in batch:
        length = len(sequence)
        padding = [SPECIAL_TOKENS["PAD"]] * (max_len - length)
        padded_sequences.append(sequence + padding)
        padded_segments.append(segment + padding)
        padded_targets.append(target + padding)
        is_nexts.append(is_next)

    count = sum([1 for target in targets for token in target if token != SPECIAL_TOKENS["PAD"]])
    padded_targets = torch.LongTensor(padded_targets)
    is_nexts = torch.LongTensor(is_nexts)
    return (padded_sequences, padded_segments), (padded_targets, is_nexts), count


class BertDataset:
    """
    Simplified construction of the BERT dataset. Included loading, tokenization, masking, special token adding.
    """
    def __init__(self, data_path, WP):
        self.WP = WP
        self.vocab = self.WP.vocab
        self.threshold = 0.15
        sentence_ids = self.load_data(data_path)
        self.sentence_pairs = self.create_sentence_pairs(sentence_ids)

    def __getitem__(self, item):
        x = random.choice(self.sentence_pairs)
        return (x["masked_sentence"], x["segment_sentence"]), (x["target_sentence"], x["is_next"])

    def __len__(self):
        return len(self.sentence_pairs)

    def load_data(self, data_path):
        with open(data_path) as data_file:
            data = data_file.read()
            clean_data = [sentence.strip().lower() for sentence in data.split("|")]
            logging.debug(clean_data)
            tokenize_data = self.WP.tokenize(clean_data)
            logging.debug(tokenize_data)
            convert_to_ids = self.WP.tokens_to_ids(tokenize_data)
            logging.debug(convert_to_ids)
            return convert_to_ids

    def create_sentence_pairs(self, sentence_ids):
        sentence_pairs = []
        for s1, s2 in zip(sentence_ids, sentence_ids[1:]):
            logging.debug(f"s1: {s1}, s2: {s2}")
            is_next = 1
            sentence_pair = {}

            if random.random() < 0.5:
                s2 = random.choice(sentence_ids)
                is_next = 0

            mask_s1 = self.mask_sentence(s1)
            mask_s2 = self.mask_sentence(s2)
            full_masked_sentence = [SPECIAL_TOKENS["CLS"]] + mask_s1 + [SPECIAL_TOKENS["SEP"]] + mask_s2 + [SPECIAL_TOKENS["SEP"]]
            full_target_sentence = [SPECIAL_TOKENS["CLS"]] + s1 + [SPECIAL_TOKENS["SEP"]] + s2 + [SPECIAL_TOKENS["SEP"]]
            full_segment_sentence = [0] + [0] * len(mask_s1) + [0] + [1] * len(mask_s2) + [1]

            sentence_pair["masked_sentence"] = full_masked_sentence
            sentence_pair["target_sentence"] = full_target_sentence
            sentence_pair["segment_sentence"] = full_segment_sentence
            sentence_pair["is_next"] = is_next
            sentence_pairs.append(sentence_pair)

        return sentence_pairs

    def mask_sentence(self, sentence):
        masked_sentences = []
        for token in sentence:
            if random.random() < self.threshold:
                r = random.random()
                if r < 0.8:
                    masked_sentences.append(SPECIAL_TOKENS["MASK"])
                elif r < 0.9:
                    masked_sentences.append(random.choice(list(self.vocab.values())[5:]))
                else:
                    masked_sentences.append(token)
            else:
                masked_sentences.append(token)
        return masked_sentences


class BertLoss(nn.Module):
    """
    Implementation of the BERT Loss which comprises of equal weighting of the MLM - CrossEntropyLoss
    and NSP - CrossEntropyLoss.
    """
    def __init__(self, model):
        super(BertLoss, self).__init__()
        self.model = model
        self.NSP = nn.CrossEntropyLoss()
        self.MLM = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["PAD"])

    def forward(self, x, y):
        """
        Computes the summed loss of the "Masked Language Model" loss and "Next Sentence Prediction" loss.
        Both are CrossEntropy Loss.
        1. Let model predict on the masked sentence --> returns a prediction for all the tokens and a next-sentence-prediction.
            1.1 tkn_pred shape: batch_size * tokens * vocab_size --> e.g. 32 * 512 * 30.000
            1.2 nxt_sntc_pred shape: batch_size * 2
        2. Take the argmax for each token and next-sentence-prediction.
            2.1 argmax is equivalent to "most likely" --> e.g. for each of the 512 tokens you get a score for each of the
            30.000 possible tokens -> simply take the token with the highest score.
            2.2 Do the same for next-sentence-prediction.
        3. Flatten both arrays to make it compatible with nn.CrossEntropyLoss().
            3.1 Instead of having batch_size * tokens * vocab_size you have (batch_size*token) * vocab_size.
            3.2 This is a 2d matrix now instead of a batched tensor.
        4. Compute both losses and sum them up.

        :param x: Masked input sequence to model. Shape: batch_size * tokens
        :param y: Unmasked input sequence and next-sentence-prediction label.
        :return: token-predictions, next-sentence-prediction and loss
        """
        y_hat = self.model(x)
        tkn_pred, nxt_sntc_pred = y_hat
        tkn_actual, nxt_sntc_actual = y

        mlm_pred, nsp_pred = tkn_pred.argmax(dim=2), nxt_sntc_pred.argmax(dim=1)

        batch_size, tokens, vocab_size = tkn_pred.size()

        tkn_pred_flat = tkn_pred.view(batch_size * tokens, vocab_size)
        tkn_actual_flat = tkn_actual.view(batch_size * tokens)

        mlm_loss = self.MLM(tkn_pred_flat, tkn_actual_flat)
        nsp_loss = self.NSP(nxt_sntc_pred, nxt_sntc_actual)

        loss = mlm_loss + nsp_loss
        return mlm_pred, nsp_pred, loss.unsqueeze(dim=0)


class PositionalEmbedding(nn.Module):
    """
    Implementation of PositionalEmbedding with adaptive size in the input sequence.
    """
    def __init__(self, max_len, dim):
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = nn.Embedding(max_len, dim)
        self.range = torch.arange(0, max_len)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = self.range[:seq_len].unsqueeze(0).repeat(batch_size, 1)
        return self.positional_embedding(positions)


class SegmentationEmbedding(nn.Module):
    """
    Segmentation Embedding used for enabling the model to distinguish between the separate sentences.
    forward() assumes batch of segments consisting of 0's and 1's --> batch_size x seq_len
    returns embedding vector for 0's and 1's --> batch_size x seq_len x dim
    """
    def __init__(self, dim):
        super(SegmentationEmbedding, self).__init__()
        self.segmentaion_embedding = nn.Embedding(2, dim)

    def forward(self, segments):
        return self.segmentaion_embedding(segments)


class Embedding(nn.Module):
    """
    BERT-specific embedding layer which uses the three parts: Token-, Segmentation-, Positional-Embedding
    with subsequent addition of the three parts.
    """
    def __init__(self, vocab, tokens=512, dim=768):
        super(Embedding, self).__init__()
        self.WordPieceBert = WordpieceTokenizer(vocab, "[UNK]")
        self.vocab_len = len(self.WordPieceBert)
        self.TokenEmbedding = nn.Embedding(len(self.WordPieceBert), dim)
        self.Segmentation = SegmentationEmbedding(dim)
        self.PositionalEmbedding = PositionalEmbedding(tokens, dim)

    def forward(self, x):
        sequences, segments = x
        sequences = torch.IntTensor(sequences)
        segments = torch.IntTensor(segments)
        embedding = self.TokenEmbedding(sequences)
        embedding += self.PositionalEmbedding(sequences)
        embedding += self.Segmentation(segments)
        return embedding


class Attention(nn.Module):
    """
    Simple Self-Attention algorithm. Potential for optimization using a non-quadratic attention mechanism in complexity.
    -> Linformer, Reformer etc.
    """
    def __init__(self, dim=768, heads=8):
        super(Attention, self).__init__()
        d = dim // heads
        self.q, self.k, self.v = nn.Linear(dim, d), nn.Linear(dim, d), nn.Linear(dim, d)
        self.norm = d ** 0.5

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        qk = torch.softmax(q @ torch.transpose(k, 1, 2) / self.norm, dim=1)
        attn = torch.matmul(qk, v)
        return attn


class MultiHeadAttention(nn.Module):
    """
    Implementation of MultiHeadAttention, splitting it up to multiple Self-Attention layers and concatenating
    the results and subsequently running it through one linear layer of same dimension.
    """
    def __init__(self, dim=768, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.self_attention_heads = nn.ModuleList([Attention() for _ in range(heads)])
        self.projector = nn.Linear(dim, dim)

    def forward(self, x):
        for i, sa_head in enumerate(self.self_attention_heads):
            if i == 0:
                out = sa_head(x)
            else:
                out = torch.cat((out, sa_head(x)), axis=-1)
        out = self.projector(out)
        return out


class Encoder(nn.Module):
    """
    Transformer encoder using MultiHeadAttention and MLP along with skip connections and LayerNorm
    """
    def __init__(self, dim=768):
        super(Encoder, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention()
        self.LayerNorm1 = nn.LayerNorm(dim)
        self.LayerNorm2 = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(*[
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU()
        ])

    def forward(self, x):
        attn = self.MultiHeadAttention(x)
        x = x.add(attn)
        x = self.LayerNorm1(x)
        mlp = self.MLP(x)
        x = x.add(mlp)
        x = self.LayerNorm2(x)
        return x


class BERT(nn.Module):
    """
    Implementation of BERT for pretraining. Easily adaptable for finetuning.
    """
    def __init__(self, N, dim, masking_rate=0.15, tokens=512, batch_size=128, vocab_file="vocab.txt"):
        super(BERT, self).__init__()
        self.dim = dim
        self.masking_rate = masking_rate
        self.tokens = tokens
        self.batch_size = batch_size

        self.Embedding = Embedding(vocab_file, self.tokens)
        self.EncoderLayers = nn.ModuleList([Encoder() for _ in range(N)])
        self.Next_Sentence_Predictor = nn.Linear(in_features=dim, out_features=2)
        self.Token_Prediction = nn.Linear(in_features=dim, out_features=self.Embedding.vocab_len)

    def forward(self, x):
        embed = self.Embedding(x)
        for enc_layer in self.EncoderLayers:
            embed = enc_layer(embed)
        nxt_sntc = self.Next_Sentence_Predictor(embed[:, 0, :])  # just take the [cls] token
        tkn_prd = self.Token_Prediction(embed)
        return tkn_prd, nxt_sntc


def pretrain(config, **kwargs):
    """
    Pretrain the Bert model given train_data and validate on val_data.
    :param train_data: Path to training data in txt file.
    :param val_data: Path to validation data in txt file.
    :param vocab_path: Path to vocabulary in txt file.
    :param batch_size: batch size for training
    :param N: Number of encoders in BERT model.
    :param dim: Latent dimension of BERT model.
    :param kwargs: Further arguments passed to BERT model.
    :return:
    """
    random.seed(0)
    torch.manual_seed(0)
    tokenizer = WordpieceTokenizer(config["vocab_path"], "[UNK]")
    train_dataset = BertDataset(config["train_data"], tokenizer)
    val_dataset = BertDataset(config["val_data"], tokenizer)

    train_data = DataLoader(train_dataset, config["batch_size"], collate_fn=padding_batch)
    val_data = DataLoader(val_dataset, config["batch_size"], collate_fn=padding_batch)

    model = BERT(config["N"], config["dim"], **kwargs)
    loss = BertLoss(model)
    optim = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1)

    for i in tqdm.tqdm(range(config["epochs"])):
        logging.info(f"Starting epoch: {i+1}")
        for batch in train_data:
            (sequence, segment), (actual, is_next), count = batch
            mlm_pred, nsp_pred, loss_value = loss((sequence, segment), (actual, is_next))

            optim.zero_grad()
            loss_value.backward()
            optim.step()

        if i % config["log_every"]:
            mlm_acc, nsp_acc = mlm_accuracy(mlm_pred, actual), nsp_accuracy(nsp_pred, is_next)
            logging.info(f"MLM-Accuracy: {mlm_acc}, NSP-Accuracy: {nsp_acc}")

            if config["log_example_pred"]:
                logging.info("Example prediction of current model:")
                random_idx = random.randint(0, len(mlm_pred)-1)
                random_pred = mlm_pred[random_idx].tolist()
                random_actual = actual[random_idx].tolist()
                random_masked = sequence[random_idx]

                random_sentence_pred = tokenizer.ids_to_tokens(random_pred)
                random_sentence_actual = tokenizer.ids_to_tokens(random_actual)
                random_sentence_masked = tokenizer.ids_to_tokens(random_masked)

                logging.info(f"Masked sentence: {random_sentence_masked}\n"
                             f"Predicted sentence: {random_sentence_pred}\n"
                             f"Actual sentence: {random_sentence_actual}")

            mlm_val, nsp_val = [], []
            for val_batch in val_data:
                (sequence, segment), (actual, is_next), count = val_batch
                mlm_pred, nsp_pred, loss_value = loss((sequence, segment), (actual, is_next))
                mlm_acc, nsp_acc = mlm_accuracy(mlm_pred, actual), nsp_accuracy(nsp_pred, is_next)
                mlm_val.append(mlm_acc)
                nsp_val.append(nsp_acc)
            logging.info(f"MLM-Accuracy-VAL: {np.mean(mlm_val)}, NSP-Accuracy-VAL: {np.mean(nsp_val)}")


if __name__ == '__main__':
    config = {"train_data": "data.txt",
             "val_data": "val_data.txt",
             "vocab_path": "vocab.txt",
             "batch_size": 1,
             "epochs": 69,
             "log_every": 10,
             "log_example_pred": True,
             "N": 6,
             "dim": 768}

    pretrain(config)

