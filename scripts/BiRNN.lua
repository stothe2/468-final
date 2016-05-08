--[[
This is an adaptation of Element-Research rnn library file "SeqBRNN.lua"
on https://github.com/Element-Research/rnn/blob/master/SeqBRNN.lua.

A Bi-direction RNN using two GRU modules. In GRU, one hidden unit consists
of two gates (update and reset), and a "candidate" hidden unit. The gates
use a sigmoid activation function, while the "candidate" hidden unit is
transformed with tanh.

The input is a tensor. Example, time x batch x inputdim.
Output is a tensor of same length. Example, time x batch x outputdim.

Reversal of sequence for backward states happens on the time dimension.

For each step, the outputs of both rnn are merged together using the merge
module (defaults to nn.CAddTable() which sums the activations).
--]]
local BiRNN, parent = torch.class('BiRNN', 'nn.Container')

function BiRNN:__init(inputSize, hiddenSize, seqLen, wordEmbeddingSize)
  parent.__init(self)

  self.wordEmbeddingSize = wordEmbeddingSize
  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()

  self.forwardModule = nn.GRU(inputSize, hiddenSize, seqLen)
  self.backwardModule = nn.GRU(inputSize, hiddenSize, seqLen)
  self.merge = nn.CAddTable() -- ??

  self.module = self:_buildModel()

  --self.w = torch.Tensor(hiddenSize, wordEmbeddingSize) -- Weight matrix for h->h
  --self.u = torch.Tensor(hiddenSize, hiddenSize) -- Weight matrix for x->h
  --self.h0 = torch.Tensor()
end

function BiRNN:_buildModel()
  -- Reverse, compute, and unreverse
  local backward = nn.Sequential()
  backward:add(nn.SeqReverseSequence(1)) -- Reverse tensor in 1D
  backward:add(self.backwardModule)
  backward:add(nn.SeqReverseSequence(1)) -- Unreverse tensor in 1D

  local concat = nn.ConcatTable()
  concat:add(self.forwardModule):add(backward)

  local birnn = nn.Sequential()
  birnn:add(concat)
  birnn:add(self.merge)

  return birnn
end

function BiRNN:updateOutput(input)
  self.output = self.module:updateOutput(input)
  return self.output
end

function BiRNN:updateGradInput(input, gradOutput)
  self.gradInput = self.module:updateGradInput(input, gradOutput)
  return self.gradInput
end

function BiRNN:accGradParameters(input, gradOutput, scale)
  self.module:accGradParameters(input, gradOutput, scale)
end

function BiRNN:__tostring__()
  if self.module.__tostring__ then
    return torch.type(self) .. ' @ ' .. self.module:__tostring__()
  else
    return torch.type(self) .. ' @ ' .. torch.type(self.module)
  end
end
