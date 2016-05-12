import optparse
import sys, os
import codecs
from nltk.tokenize import TweetTokenizer
import nltk.tokenize.api
import tinysegmenter

class myTinySegmenter(tinysegmenter.TinySegmenter, nltk.tokenize.api.TokenizerI):
    pass


baseline_path = os.getcwd()[:-36]
if baseline_path not in sys.path:
    sys.path.insert(0, baseline_path)

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="data", default=baseline_path+"en-jp-gold/bitext/parallel.gold.en-jp", help="Parallel corpus")
(opts, _) = optparser.parse_args()

ref = [pair.split(' ||| ') for pair in open(opts.data)]
num_sents = len(ref)

tknzr = TweetTokenizer()
segmenter = myTinySegmenter()

en_tokenized = codecs.open("bitext.en.tok.txt", "w", "utf-8")
ja_tokenized = codecs.open("bitext.ja.tok.txt", "w", "utf-8")

for pair in ref:
    en_tokenized.write(" ".join(tknzr.tokenize(pair[0])))
    en_tokenized.write("\n")
    ja_tokenized.write(" ".join(segmenter.tokenize(pair[1].decode("utf-8"))))
    ja_tokenized.write("\n")

en_tokenized.close()
ja_tokenized.close()
