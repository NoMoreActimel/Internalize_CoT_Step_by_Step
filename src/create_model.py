import torch
import os

from src.model import ImplicitModel
from src.model_jepa import JEPAImplicitModel
from src.configuration_model import ImplicitModelConfig
from src.utils import expand_gpt2_positions

def create_model(args, device):
    if args.from_pretrained is None:
        config = ImplicitModelConfig(base_model=args.model, n_head=args.n_head)
        model = ImplicitModel(
            config,
            reinitialize_weights=args.train_from_scratch,
            use_flash_attention=args.flash_attention_2,
            use_peft=args.use_peft
        ).to(device)
    else:
        print (f'Loading from {args.from_pretrained}')
        model = ImplicitModel.from_pretrained(
            args.from_pretrained,
            use_flash_attention=args.flash_attention_2,
            use_peft=args.use_peft
        ).to(device)
        config = model.config

    if 'gpt2' in args.model:
        expand_gpt2_positions(model, args)
    
    model = model.to(device)
    tokenizer = model.tokenizer

    if args.reinitialize_weights:
        print ('reinitializing weights')
        model.model.apply(model.model._init_weights)

    if args.keep_position:
        assert 'gpt2' in args.model # only implemented for gpt2 generate TODO: the code for this is not checked in yet
    
    return config, model, tokenizer

def create_jepa_model(config, ref_model, args, device):
    if args.from_pretrained is None:
        model = JEPAImplicitModel(
            config=config,
            ref_model=ref_model,
            alpha_logits_loss=args.alpha_logits_loss,
            ref_model_update_decay=args.ref_model_update_decay,
            logits_loss_on_full_cot=args.logits_loss_on_full_cot,
            reinitialize_weights=args.train_from_scratch,
            use_flash_attention=args.flash_attention_2,
            use_peft=args.use_peft
        ).to(device)
    else:
        print (f'Loading from {args.from_pretrained}')
        model = JEPAImplicitModel(
            config=config,
            ref_model=ref_model,
            alpha_logits_loss=args.alpha_logits_loss,
            ref_model_update_decay=args.ref_model_update_decay,
            logits_loss_on_full_cot=args.logits_loss_on_full_cot,
            reinitialize_weights=False,
            use_flash_attention=args.flash_attention_2,
            use_peft=args.use_peft
        ).to(device)
        state_dict = torch.load(os.path.join(args.from_pretrained, 'state_dict.bin'))
        model.load_state_dict(state_dict, strict=True)

    if 'gpt2' in args.model:
        expand_gpt2_positions(model, args)
    
    if args.keep_position:
        assert 'gpt2' in args.model # only implemented for gpt2 generate TODO: the code for this is not checked in yet
    
    return config, model, model.tokenizer
