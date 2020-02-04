from textgenrnn import textgenrnn

def create_textgen(model_name):
    return textgenrnn(weights_path='{}_weights.hdf5'.format(model_name),
                     vocab_path='{}_vocab.json'.format(model_name),
                     config_path='{}_config.json'.format(model_name),
                     name=model_name)
t = create_textgen('v1_no_context')
t.generate(interactive=True, top_n=5)
