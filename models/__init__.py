from .cl_lm import ConceptualLM
chose_model = {'cllm':ConceptualLM}
def get_model(name, config):
    return chose_model[name](config)