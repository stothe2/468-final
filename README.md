## Implementing a Neural Network Model for Translation of Japanese-English Tweets

**—Collaborated with [Sarah Watanabe](https://github.com/swatana3).**

Currently in Machine Translation, translating social media text is a challenge. User generated content (UGC) is highly noisy (spam, ads), domain unrestricted (anyone anywhere can be there), user-centric (users are given more flexiblilty and choices), generated in high volume, and focused on knowledge and context sharing at the expense of grammatical, spelling, and other linguistic errors.

Therefore, our challenges lies in being able to create a machine translation system that

1. is large-scale and as close to real-time as possible in
data management,
2. will preserve the meaning of words, and
3. will handle errors in linguistics and in canonical writing (verbs, grammers, typos, wrong punctuation, unstructured syntax, etc.).


## Neural networks for translation

Neural-based machine translation research dates back to Forcada and Neco (1997). While traditional Statistical Machine Translation (SMT) models rely on pre-designed features (like POS tags, etc.), neural machine translation models do not make use of any pre-designed features. That is to say, all features they learn are from training, and this maximizes their performance.

Recently proposed NMT models, like those by Kalchbrenner and Blunsom (2013), Sutskever *et al.* (2014), Cho *et al.* (2014), and Bahdanau *et al.* (2015) have showed comparable performace to state-of-the-art SMT models like Moses (Koehn et al., 2003). The base of all these neural-based models is [mostly] the *encoder-decoder* architecture. A variable-length input is encoded as a fixed-length vector, which is then decoded to a variable-length output (Sutskever *et al.*, 2014; Cho *et al.*, 2014). The hidden state *h* is where all the magic of translation happens.

In this simple encoder-decoder architecture, one is essentially cramming information of an entire sentence into a single vector. This is not reasonable, and indeed, it has been shown that as sentence length increases, the performance of the neural network degrades (Cho *et al.*, 2014).

We will survey the current state-of-the-art neural network architecture (Bahdanau *et al.*, 2015), and use it to translate tweets from Japanese to English.

## Model

The current state-of-the-art NMT model (RNNSearch) makes use of a **Bidirectional RNN (BiRNN)** to encode the input `x` to a *sequence of vectors*, of which a *subset* is chosen during translation by the **Gated Recursive Unit (GRU)** decoder.

We won't go into depth to describe Recursive Neural Networks (RNN) here (see [this](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) blog post by Andrej Karpathy or [this](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) tutorial by Denny Britz for well-written introductions), but only focus on the RNN extensions used in the chosen model.

#### The BiRNN Encoder

While in a vanilla RNN the output `y` is dependent on current input `x` and all previous inputs, a bidirectional RNN assumes that `y` is not only dependent on preceding inputs but also on forward/upcoming inputs. We first compute the *forward states* $\overrightarrow{h_t}$ by iterating over the sentence $x = (x_1,...,x_{T_x})$. Then, we compute the backward states $\overleftarrow{h_t}$ by iterating over the reverse of the same sentence $x = (x_{T_x},...,x_1)$. Finally, we concatenate the two to obtain the states $(h_1,h_2,...h_{T_x})$ where

$$ h_t =
\begin{bmatrix}
\overrightarrow{h}_{t}\\
\overleftarrow{h}_{t}
\end{bmatrix} $$

$h_t$ is computed as

$$ h_t =
\begin{cases}
(1 - z_t) * h_{t-1} + z_t * \tilde{h_t} & \text{ if } t > 0 \\
0 & \text{ if } t = 0
\end{cases} $$

As you can see, there are two gates: the *update gate* $z_t$ and the *reset gate* $r_t$. Also, `*` represents element-wise operation; and

$$ \tilde{h_t} = \tanh (W E x_t + U [ r_t * h_{t-1} ]) $$

$$ z_t = \sigma (W_z E x_t + U_z h_{t-1} ) $$

$$ r_t = \sigma (W_r E x_t + U_r h_{t-1} ) $$

Let *m* be the word embedding dimensionality or sentence size, *n* be the number of hidden states, and $ K_x $ and $ K_y $ be the `vocabulary_size` for source and target languages, respectively (note, in our implementation $ K_x = K_y $). Then, $ E \in \mathbb{R}^{m \times K_x} $ is the word embedding matrix, and $ W, W_z, W_r \in \mathbb{R}^{n \times m} $ and $ U, U_z, U_r \in \mathbb{R}^{n \times n} $ are the weight matrices.

Its important to note that only the word embedding matrix *E* is shared between the forward and backward states (none of the other matrices or gates are shared).

#### GRU Decoder

The Gated Recursive Unit (GRU) was proposed by Cho *et al.* (2014). We won't desribe it in detail here, but see [this](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) post by Christopher Olah to understand the reasoning behind them. (will post more on this later).

## Training: SGD and BPTT

As by now you would know, neural networks start out "blank". That is to say, they make use of no pre-designed feature parameters. The weight matrices are [generally] randomly initialized. It is the goal of training to find matrices that give rise to most desirable behavior (loss function).

We use Stochastic Gradient Descent (SGD) algorithm to minimize the error loss.  What it does is that it trains the pamareters of the neural network to move in a direction that minimizes error. This direction is given by the gradients on the loss.


## Experiment Settings

#### Data

Our dataset is the Japanese-English parallel corpus on [microtopia](http://www.cs.cmu.edu/~lingwang/microtopia/). We used 100% of the data for training because the corpus only had 953 sentence pairs.

We found only 2831 unique words in the English text, and 1777 unique words in the Japanese text. This means our `vocabulary_size` size is limited to those numbers. Any word not in our vocabulary is mapped to `UNK`. For example, say "Johns" in an infrequent word in our training corpus. Then the sentence "Johns Hopkins University is in Baltimore" will be processed as "UNK Hopkins University is in Baltimore".

#### Code

Our initial plan was to use [Theano](http://deeplearning.net/software/theano/) and [Blocks](http://blocks.readthedocs.org/en/latest/). However, compared to Theano, we found [Torch](http://torch.ch/) more user-friendly and easy-to-pick-up.

We could only write an implementation of a BiRNN with Torch, and got confused with the usage of layers (we kind of got the gist of how it'd be implemented for a classifying task, or even for language model training, but the implementing it in as part of an encoder-decoder architecture for machine translation confused us). In the end, we decided to switch base and use [original code](https://github.com/lisa-groundhog/GroundHog/tree/master/experiments/nmt) used in Bahdanau *et al.*. Note, however, their code makes use of GroundHog which is now deprecated and, thus, not ideal. But for our purposes, it works well.

## File Structure

The main files reside in the `GroundHog/experiment/nmt` directory. We make use of the following files:
- `state.py` defines all parameters used by the model
- `encdec.py` main model
- `train.py` training code

In addition, data processing files are in the `GroundHog/experiment/nmt/preprocess` directory. We used all the original scripts to process the data, and in addition wrote our own tokenizer `tokenize.py`.

Our output from the training run is placed in the main directory, and called `output`.

## Usage

#### Preprocessing

`cd` into `GroundHog/experiment/nmt/preprocess` and start by tokenizing the parallel courpus.
```
$ python tokenizer.py
```
Then, run the following commands.
```
$ python preprocess.py bitext.en.tok.txt -d vocab.en.pkl -v 30000 -b binarized_text.en.pkl -p
$ python invert-dict.py vocab.en.pkl ivocab.en.pkl
$ python convert-pkl2hdf5.py binarized_text.en.pkl binarized_text.en.h5

$ python preprocess.py bitext.ja.tok.txt -d vocab.ja.pkl -v 30000 -b binarized_text.ja.pkl -p
$ python invert-dict.py vocab.ja.pkl ivocab.ja.pkl
$ python convert-pkl2hdf5.py binarized_text.ja.pkl binarized_text.ja.h5

$ python shuffle-hdf5.py binarized_text.en.h5 binarized_text.ja.h5 binarized_text.en.shuf.h5 binarized_text.ja.shuf.h5
```

Files `binarized_text.en.shuf.h5` and `binarized_text.ja.shuf.h5` are what we use as inputs to our model. However, due to some bug in the code, you might have to rename them in a single letter, and place it in the same directory as the scripts, that is `GroundHog/experiment/nmt`. We called renamed `binarized_text.en.shuf.h5` to `s` and `binarized_text.ja.shuf.h5` to `t`.

#### Training

```
$ THEANO_FLAGS='floatX=float32' python train.py --proto=prototype_encdec_state "target='t', source='s', indx_word='preprocess/ivocab.en.pkl', indx_word_target='preprocess/ivocab.ja.pkl', word_indx='preprocess/vocab.en.pkl', word_indx_trgt='preprocess/vocab.ja.pkl', null_sym_source=2831, null_sym_target=1777, n_sym_source=2832, n_sym_target=1778, allow_input_downcast=True, timeStop=60"
```

The output from this is in the main directory, called `output`.

## Results

See the `Human_Eval_*` files for the analysis.

## Resources

##### Neural-based MT
1. [Neural Machine Translation by Jointly Learning to Align and Translate](http://arxiv.org/pdf/1409.0473.pdf). Bahdanau *et al.* (2015).
2. [Neural Machine Translation of Rare Words with Subword Units](http://arxiv.org/pdf/1508.07909v3.pdf). Sennrich *et al.* (2015).
3. [Character-Aware Neural Language Models](http://arxiv.org/pdf/1508.06615.pdf). Kim *et al.* (2015).
4. [A Character-Level Decoder without Explicit Segmentation for Neural Machine Translation](http://arxiv.org/pdf/1603.06147.pdf). Chung *et al.* (2016).
5. [Recurrent Neural Networks Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
6. [Deep Learning with Torch Tutorial](https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb)
7. [Theano Tutorial](http://nbviewer.jupyter.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb)

##### Works usings Twitter datasets
1. [Automatic Keyword Extraction on Twitter](http://www.cs.cmu.edu/~lingwang/papers/acl2015-3.pdf). Ling *et al.* (2015).

##### Lua
1. [Reference](http://lua-users.org/files/wiki_insecure/users/thomasl/luarefv51.pdf)
2. [Learn Lua in 15 minutes](http://tylerneylon.com/a/learn-lua/)
