Pytorch implementation of Bidirectional Encoder Representation Transformer (BERT)
---------------------------------------------------------------------------------
Bert is simply a bunch of Transformer Encoder stacked on each other, just like Vision Transformer.
The main invention here is that it learns state of the art representations for language due to 2
specific pretraining mechanisms, which makes it super easy to finetune on specific / downstream tasks.
It's a sequence to sequence (seq2seq) model, meaning it takes as input a sequence and outputs a sequence too.
A sequence is nothing like a sentence, however the sentence was embedded before. In Bert's case the embedding
consists of 3 steps. First every word is being mapped / embedded to a high dimensional vector after using WordPiece as the Tokenizer.
We add a learned segmentation to this embedding. The intention is that Bert should be able to distinguish between
different sentences in the input. This is necessary for Question and Answer inputs. As a result, Bert learns two
segmentation embeddings for the Question and Answer, respectively. Third a standard positional encoding is also
added to the token, as also used in the original Transformer paper. The necessity for this is due to the fact
that Transformers, more specifically the attention mechanisms, are "set-operations". This means that in the attention
mechanism every token attends to every token without having an sense of order. Therefore, the sentence
"Anna gave Dome a gift" can't be distinguished from "Dome gave Anna a gift". To overcome this issue, we simply
add positional encodings to the token.

input_token = token + segmentation + position <-- where token is the embedded word

After every word in our input sentence was properly embedded, we add a [cls] token to the beginning of the
sequence. This [cls] token will be the "aggregated sequence representation" for classification tasks.
For example, in a task to predict the emotion of the writer the [cls] token will contain the summarized
information of the mood of the writer's sentence. This can then be fed through a simple FeedForward to get a
classification over the sentence's emotion: happy or sad.
If we have 2 sentences we add a [sep] token between the two sentences.

### Pre-training mechanisms
1. #### Masked Language Modeling (MLM)
   1. to make the "bidirectionality" work, the authors use masking.
   2. 15% of the input tokens get masked out
   3. replace the masked token with:
      1. in 80% of the cases with [mask]
      2. in 10% of the cases with a random token
      3. in 10% don't change anything
   4. the latter has the purpose that the Transformer has to learn the context to predict masked words or find replaced words
   5. the final representation of every masked token will be used as the prediction for the original token

2. #### Next Sentence Prediction (NSP)
   1. A simple binary task to predict whether sentence B is actually coming after sentence A
   2. Again the [cls] token is the prediction for this binary task

### Architecture
1. Sentence
2. Embedding (tokenization + segmentation + positional encoding)
3. Nx Transformer Encoder
   1. Multi Head Attention
   2. Add Residual Connection from Embedding
   3. Layer Normalization
   4. MLP / FeedForward
      1. Linear
      2. GELU
      3. Linear
      4. GELU
   5. Add Residual Connection from ii.
   6. Layer Normalization
4. Classification layer ([cls] token into output layer for classification or all token representations for token-level tasks)

### Loss
The goal is to minimize both the MLM and NSP objectives together.
The MLM objective is the cross-entropy loss between the original token and the predicted token.
The NSP also uses the cross-entropy loss between original label and the predicted label.

### Useful Ressources
- https://github.com/dreamgonfly/BERT-pytorch
- BERT Embedding: https://medium.com/@_init_/why-bert-has-3-embedding-layers-and-their-implementation-details-9c261108e28a
```bibtex
@article{DBLP:journals/corr/abs-1810-04805,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
