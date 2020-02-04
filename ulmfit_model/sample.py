from fastai.text import *

# load data_lm 
data_lm = TextLMDataBunch.load(path, 'leaves-of-ai/app/static/data_lm', bs=bs)
# load learner
