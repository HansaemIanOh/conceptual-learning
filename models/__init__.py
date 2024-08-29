from .simpletransformer import SimpleTransformer
chose_model = {'simpletransformer':SimpleTransformer,
               }
def get_model(name, config):
    return chose_model[name](config)