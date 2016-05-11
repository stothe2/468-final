require 'dp'
require 'paths'

local function trim(s)
  -- from PiL2 20.4
  return (s:gsub('^%s*(.-)%s*$', '%1'))
end

local function spairs(t, order)
  -- from http://stackoverflow.com/questions/15706270/sort-a-table-in-lua

  -- Collect the keys
  local keys = {}
  for k in pairs(t) do keys[#keys+1] = k end

  -- If order function given, sort by it by passing the table and keys a, b
  -- Otherwise just sort the keys
  if order then
    table.sort(keys, function(a,b) return order(t, a, b) end)
  else
    table.sort(keys)
  end

  -- Return the iterator function
  local i = 0
  return function()
    i = i + 1
    if keys[i] then
      return keys[i], t[keys[i]]
    end
  end
end

local function _process1()
  local reference = string.gsub(paths.cwd(), 'scripts', '') .. 'en-jp-gold/bitext/parallel.gold.en-jp'

  local en = {}
  local ja = {}
  for line in io.open(reference, 'r'):lines() do
    pair = string.split(line, ' ||| ')
    table.insert(en, trim(pair[1])) -- Fix trim
    table.insert(ja, trim(pair[2])) -- Fix trim
  end
  return en, ja
end

local function _process2()
  local enRef = string.gsub(paths.cwd(), 'scripts', '') .. 'en-ja.txt/OpenSubtitles2016.en-ja.en'
  local jaRef = string.gsub(paths.cwd(), 'scripts', '') .. 'en-ja.txt/OpenSubtitles2016.en-ja.ja'

  local en = {}
  local ja = {}
  for line in io.open(enRef, 'r'):lines() do
    line = string.gsub(line, '[$%-]+', '') -- Fix special case "- I will go." to " I will go."
    table.insert(en, trim(line))
  end
  for line in io.open(jaRef, 'r'):lines() do
    line = string.gsub(line, '[$%-]+', '') -- Fix special case "- I will go." to " I will go."
    table.insert(ja, trim(line))
  end
  return en, ja
end

function collectdata(dataType)
  if dataType == "microtopia" then
    return _process1()
  elseif dataType == "opensub" then
    return _process2()
  else
    error("Invalid data option")
  end
end

function buildvocab(data)
  vocab = {}
  for sIndex, s in pairs(data) do
    words = string.split(string.lower(s), '%s+')
    for wIndex, w in pairs(words) do
      --print(w:gsub('[%p%d]', ''))
      tokens = string.split(w, '[%p%d]')
      for tIndex, t in pairs(tokens) do
        vocab[t] = vocab[t] and vocab[t]+1 or 1
      end
    end
  end

  -- Sort in descending order of word frequency, and select top 30,000 most frequent words
  mostFreqWords = {}
  count = 0
  for k, v in spairs(vocab, function(t, a, b) return t[a] > t[b] end) do
    table.insert(mostFreqWords, k)
    count = count + 1
    if count == 30000 then break end
  end
  return mostFreqWords
end

--local en, ja, ref = collectdata()
en, ja = collectdata("opensub")
print(buildvocab(en))
