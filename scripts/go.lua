require 'nn'
require 'rnn'

require 'prepare'
require 'BiRNN'

opt = {
  vocabSize = 30000, -- Number of words in our vocabulary

  batchSize = 64, -- Number of sequences to train on in parallel
  seqLen = 5, -- Sequence length: BPTT for this many time-steps
  hiddenSize = 1000, -- Number of hidden units used as output of each recurrent layer

  trainFrac = 0.95, -- Fraction of data that goes into training set
  validFrac = 0.05, -- Fraction of data that goes into validation set

  wordEmbeddingSize = 620 -- Number of rows in the word embedding matrix E (typically 100-500)

  learningRate = 0.01 -- How big of a step we want to take in each iteration for SGD
}

-- Data
--ds = dp.PennTreeBank{recurrent=true, context_size=opt.seqLen+1, bidirectional=true} -- Why is context length +1?
--trainSet = ds:trainSet()
--validSet = ds:validSet()
--testSet = ds:testSet()
en, ja = collectdata('opensub')

local birnn = BiRNN(10, 5, 2, opt.wordEmbeddingSize)
print(birnn)

for k, v in pairs(obj) do
  print(k, v)
end
