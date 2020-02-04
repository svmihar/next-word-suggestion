import pandas as pd
import pickle
from fastai.text import *
import torch

import warnings
warnings.filterwarnings("ignore")
torch.cuda.set_device(0)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)

# open file
df = pd.read_csv('../../data/listing.csv')
df.dropna(inplace=True, subset=['Title'])

# create databunch
nrows, ncols = df.shape
train_size = math.floor(nrows * 0.8)
data_lm = TextLMDataBunch.from_df(
    '.', df.iloc[:train_size], df.iloc[train_size:], text_cols=['Title'])
data_lm.save('./lm_databunch_title')
print('big databunch created')

"""
# create 5000 only databunch
df = df[:5000]
nrows, ncols = df.shape
train_size = math.floor(nrows * 0.8)
data_lm = TextLMDataBunch.from_df(
    '.', df.iloc[:train_size], df.iloc[train_size:], text_cols=['clean_paragraf'])
data_lm.save('lm_databunch_kompas_kecil_5000')
print('small databunch created')
"""

# load databunch 
def load_databunch(path): 
    bs = 32 
    path = Path('.')
    return load_data(path, 'lm_databunch_kompas', bs=bs)
print('now loading databunch object')
#data_lm = load_databunch('.')


# create learner object
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

# find optimum learning rate
learn.lr_find(start_lr=1e-10, end_lr=1e10)
learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr

# fir one cycle
# Clear unused cache https://pytorch.org/docs/stable/cuda.html#torch.cuda.empty_cache
torch.cuda.empty_cache()
learn.fit_one_cycle(cyc_len=1, max_lr=slice(min_grad_lr,min_grad_lr/10), moms=(0.8, 0.7))

# unfreeze model and fine-tune it
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(skip_end=15, suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr

# fit x epoch
learn.fit_one_cycle(cyc_len=300, max_lr=slice(min_grad_lr*10, min_grad_lr/10), moms=(0.9, 0.88888888))
learn.save_encoder('./ft_enc_title')
torch.save(learn.model.state_dict(), './lm_encoder.pth')

# predict words
TEXT = "cuci ac"
N_WORDS = 7


def compose(TEXT, N_WORDS, temp):
    composed = learn.predict(TEXT, N_WORDS, temperature=temp)
    return composed

def compose_beamsearch(TEXT, N_WORDS, temp):
    composed = learn.beam_seach(TEXT, N_WORDS, temperature=temp)
    return composed
for a in [.2,.5,.75,1]:
    for _ in range(10): 
        print(compose(TEXT, N_WORDS, a))
