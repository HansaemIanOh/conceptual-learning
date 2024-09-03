from .cl_cac_bert import CLCAC
from .cl_cm_gpt import CLCM
from .cl_lm import ConceptualLM
chose_model = {'clcac':CLCAC,
               'clcm':CLCM,
               'cllm':ConceptualLM,
               }
def get_model(name, config):
    return chose_model[name](config)