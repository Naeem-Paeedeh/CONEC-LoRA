from models.conec_lora import Learner


def get_model(model_name, args):
    name = model_name.lower()
    options = {
        "conec_lora": Learner,
    }
    return options[name](args)
