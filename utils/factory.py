def get_model(model_name, args):
    name = model_name.lower()

    # The imports are lazy so that a model's dependencies are only required when
    # that model is actually used.
    if name == "conec_lora":
        from models.conec_lora import Learner
    elif name == "cllora":
        # CL-LoRA
        from models.cllora import Learner
    else:
        raise NotImplementedError("Unknown model name: {}".format(model_name))

    return Learner(args)
