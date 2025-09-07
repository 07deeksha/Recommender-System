from build_vocab import Vocabulary
import pickle
with open(r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    print("done")

with open(r'C:\Users\Deeksha\Desktop\TrendPulse\Tp\decoder-140-1.ckpt', 'rb') as f:
    vocab = pickle.load(f)
    print("ok")