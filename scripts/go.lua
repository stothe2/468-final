require 'prepare'
require 'encdec'

torch.manualSeed(1234)

opt = {
  markEnd = 30000, -- End of sequence mark
  vocabSize = 30000 + 1, -- Number of words in our vocabulary
  unknownPlaceholer = "UNKOWN_TOKEN", -- Token used to represent unknown words in text

  batchSize = 80, -- Number of sequences to train on in parallel
  rho = 5, -- BPTT for this many time-steps
  seqLen = 30, -- Maximum sentence length

  hiddenSize = 1000, -- Number of hidden units used as output of each recurrent layer
  wordEmbeddingSize = 620, -- Number of rows in the word embedding matrix E (typically 100-500)

  trainFrac = 0.95, -- Fraction of data that goes into training set
  validFrac = 0.05, -- Fraction of data that goes into validation set

  learningRate = 0.01, -- How big of a step we want to take in each iteration for SGD

  bitext = string.gsub(paths.cwd(), 'scripts', '') .. 'en-jp-gold/bitext/parallel.gold.en-jp',
  sourceVocab = nil,
  targetVocab = nil,

  computeAlignment = false
}

-- Data
--ds = dp.PennTreeBank{recurrent=true, context_size=opt.seqLen+1, bidirectional=true} -- Why is context length +1?
--trainSet = ds:trainSet()
--validSet = ds:validSet()
--testSet = ds:testSet()
--en, ja = collectdata('microtopia')

--local birnn = BiRNN(10, 5, 2, opt.wordEmbeddingSize)
local encdec = EncoderDecoder(opt.source, opt.target, opt.sourceVocab,
    opt.targetVocab, opt.markEnd, opt.vocabSize, opt.batchSize,
    opt.hiddenSize, opt.wordEmbeddingSize, opt.computeAlignment)
