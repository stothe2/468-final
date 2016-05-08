require 'nn'
require 'rnn'

require 'prepare'
require 'BiRNN'

opt = {
  vocabSize = 30000, -- Number of words in our vocabulary

  batchSize = 32, -- Number of sequences to train on in parallel
  seqLen = 5, -- Sequence length: BPTT for this many time-steps
  hiddenSize = 1000, -- Number of hidden units used as output of each recurrent layer

  trainFrac = 0.95, -- Fraction of data that goes into training set
  testFrac = 0.05, -- Fraction of data that goes into test set

  wordEmbeddingSize = 620 -- Number of rows in the word embedding matrix E (typically 100-500)
}

-- Data
ds = dp.PennTreeBank{recurrent=true, context_size=5}
trainSet = ds:trainSet()
en, ja, ref = collectdata()

local birnn = BiRNN(10, 5, 2, opt.wordEmbeddingSize)
print(birnn)
