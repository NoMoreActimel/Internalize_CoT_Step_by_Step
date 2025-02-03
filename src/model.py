import os
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList
import sys

from configuration_model import ImplicitModelConfig
from utils import get_sep_position, DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor


class ImplicitModel(nn.Module):
    def __init__(self, config, reinitialize_weights=False):
        super().__init__()
        self.config = config
        self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model, trust_remote_code=True)
        if reinitialize_weights:
            print ('Reinitializing model weights!')
            self.base_model.apply(self.base_model._init_weights)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def forward(self, input_ids, position_ids=None, output_attentions=False):
        if position_ids is not None:
            outputs = self.base_model.forward(input_ids=input_ids, output_attentions=output_attentions, position_ids=position_ids)
        else:
            outputs = self.base_model.forward(input_ids=input_ids, output_attentions=output_attentions)
        return outputs

    def compute_loss(self, input_ids, labels, position_ids=None, output_attentions=False):
        outputs = self.forward(input_ids=input_ids, position_ids=position_ids, output_attentions=output_attentions)
        logits = outputs.logits

        labels_pred = logits.argmax(-1)
        mask = labels[...,1:].ge(0)
        correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        outputs.loss = loss
        outputs.token_accuracy = token_accuracy
        outputs.total_correct = correct_tokens
        outputs.total_loss = loss * total_tokens
        outputs.total_tokens = total_tokens
        return outputs

    def generate(
            self,
            input_ids,
            max_new_tokens=512,
            num_beams=1,
            stop_on_two_eos=True,
            position_ids=None,
            use_new_tokens=False,
            new_tokens_start_id=None
    ):
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        batch_size = input_ids.shape[0]

        # Since there's one eos after CoT and another after final answer, we need to wait for two eos
        generation_config = GenerationConfig.from_model_config(self.base_model.config)
        if hasattr(generation_config, 'pad_token_id'):
            generation_config.pad_token_id = None #TODO: this might not be necessary
        if stop_on_two_eos:
            generation_config.eos_token_id = -1
            logits_processor = LogitsProcessorList([DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)])
            stopping_criteria = StoppingCriteriaList([DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)])
        else:
            logits_processor = None
            stopping_criteria = None

        if sep_positions.eq(sep_positions[0]).all():
            input_ids = input_ids[:, :sep_positions[0]+1]
            if position_ids is not None:
                position_ids = position_ids[:, :sep_positions[0]+1]
            
            beam_output = self._generate(
                input_ids, position_ids, generation_config, max_new_tokens, num_beams,
                logits_processor, stopping_criteria, use_new_tokens, new_tokens_start_id
            )
            beam_output = beam_output.unsqueeze(1)
        else:
            beam_output = []
            for i in range(batch_size):
                input_ids_i = input_ids[i:i+1]
                sep_positions_i = sep_positions[i:i+1]
                input_ids_i = input_ids_i[:, :sep_positions_i+1]
                position_ids_i = position_ids[i:i+1, :sep_positions_i+1] if position_ids is not None else None

                beam_output_i = self._generate(
                    input_ids_i, position_ids_i, generation_config, max_new_tokens, num_beams,
                    logits_processor, stopping_criteria, use_new_tokens, new_tokens_start_id
                )
                beam_output.append(beam_output_i)
        return beam_output

    def _generate(
            self,
            input_ids,
            position_ids,
            generation_config,
            max_new_tokens,
            num_beams,
            logits_processor,
            stopping_criteria,
            use_new_tokens=False,
            new_tokens_start_id=None
    ):
        if use_new_tokens:
            return self._generate_with_new_tokens(
                input_ids,
                position_ids,
                max_new_tokens,
                stopping_criteria,
                new_tokens_start_id,
                decode=False
            )
        if position_ids is not None:
            beam_output = self.base_model.generate(
                input_ids=input_ids,
                position_ids=position_ids,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                num_return_sequences=1,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
            )
        else:
            beam_output = self.base_model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                num_return_sequences=1,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
            )
        return beam_output

    def _generate_with_new_tokens(
            self,
            input_ids,
            position_ids,
            max_new_tokens,
            stopping_criteria,
            new_tokens_start_id,
            decode=False
    ):
        batch_size = input_ids.shape[0]
        start_id_tensor = torch.full((batch_size, 1), new_tokens_start_id, dtype=torch.long).to(input_ids.device)
        eos_count = torch.zeros(batch_size, dtype=torch.long).to(input_ids.device)
        
        input_ids_generated = input_ids
        position_ids_generated = position_ids
        with torch.no_grad():
            for _ in range(0, max_new_tokens, 2):
                outputs = self.base_model(input_ids=input_ids_generated, position_ids=position_ids_generated)
                next_token_logits = outputs.logits[:, -1, :]

                next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids_generated = torch.cat([input_ids_generated, next_token_ids, start_id_tensor], dim=1)

                eos_count[next_token_ids.squeeze(1) == self.tokenizer.eos_token_id] += 1
                if stopping_criteria is not None and (eos_count >= 2).all():
                    break
                elif (eos_count >= 1).all():
                    break
                
                if position_ids_generated is not None:
                    new_position_ids = torch.arange(
                        input_ids_generated.shape[1] - 2, input_ids_generated.shape[1]
                    ).unsqueeze(0).expand(batch_size, -1)
                    position_ids_generated = torch.cat([position_ids_generated, new_position_ids], dim=1)

        if not decode:
            return input_ids_generated
    
        generated_sep_positions = get_sep_position(input_ids_generated, self.tokenizer.eos_token_id, 2)
        texts_generated = []

        for i, input_ids_i in enumerate(input_ids_generated):
            input_ids_i = input_ids_i[:generated_sep_positions[i]]                              # SHOULD WE CUT ?
            texts_generated.append(self.tokenizer.decode(input_ids_i, skip_special_tokens=True))
        
        return input_ids_generated, texts_generated

    @classmethod
    def from_pretrained(self, pretrained_path):
        config = ImplicitModelConfig.from_pretrained(pretrained_path)
        model = ImplicitModel(config)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'))
        model.load_state_dict(state_dict, strict=True)
        return model

    def save_pretrained(self, save_directory):
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))
