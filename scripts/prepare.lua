require 'dp'
require 'paths'

local function trim(s)
  -- from PiL2 20.4
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

function collectdata()
  -- Load senetences into input and target Tensors
  local reference = string.gsub(paths.cwd(), 'scripts', '') .. 'en-jp-gold/bitext/parallel.gold.en-jp'

	en = {}
	ja = {}
	ref = {}
	enVocab = {} -- TODO English vocabulary
	jaVocab = {} -- TODO Japanese vocabulary
  for line in io.open(reference, 'r'):lines() do
		pair = string.split(line, ' ||| ')
    table.insert(en, trim(pair[1])) -- Fix trim
		table.insert(ja, trim(pair[2])) -- Fix trim
		table.insert(ref, {trim(pair[1]), trim(pair[2])})
  end
	return en, ja, ref
end
