import os
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList
import sys

from configuration_model import ImplicitModelConfig
from utils import get_sep_position, DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor


class ImplicitModel(nn.Module):
    def __init__(self, config, reinitialize_weights=False, use_flash_attention=False):
        super().__init__()

        self.config = config
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            trust_remote_code=True
            # attn_implementation="sdpa" if not use_flash_attention else "flash_attention_2"
        )
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

    def compute_loss(self, input_ids, labels, position_ids=None, output_attentions=False, **kwargs):
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
            new_tokens_start_id=None,
            inputs_with_cot=True,
            position_ids_shift=None
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
            if inputs_with_cot:
                input_ids = input_ids[:, :sep_positions[0]+1]
                if position_ids is not None:
                    position_ids = position_ids[:, :sep_positions[0]+1]
            
            beam_output = self._generate(
                input_ids, position_ids, generation_config, max_new_tokens, num_beams,
                logits_processor, stopping_criteria, use_new_tokens, new_tokens_start_id,
                position_ids_shift
            )
            if isinstance(beam_output, torch.Tensor):
                beam_output = beam_output.unsqueeze(1)
        else:
            beam_output = []
            for i in range(batch_size):
                input_ids_i = input_ids[i:i+1]
                position_ids_i = position_ids[i:i+1] if position_ids is not None else None
                if inputs_with_cot:
                    sep_positions_i = sep_positions[i:i+1]
                    input_ids_i = input_ids_i[:, :sep_positions_i+1]
                    position_ids_i = position_ids_i[:, :sep_positions_i+1] if position_ids is not None else None

                beam_output_i = self._generate(
                    input_ids_i, position_ids_i, generation_config, max_new_tokens, num_beams,
                    logits_processor, stopping_criteria, use_new_tokens, new_tokens_start_id,
                    position_ids_shift
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
            new_tokens_start_id=None,
            position_ids_shift=None
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
            if position_ids_shift is not None:
                # Used in case of left_to_right_removal.
                # We need to shift position ids by the length of COT = position_ids_shift
                beam_output = self._generate_with_shifted_position_ids(
                    input_ids,
                    position_ids,
                    max_new_tokens,
                    stopping_criteria,
                    position_ids_shift
                )
            else:
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

    def _generate_with_shifted_position_ids(
            self,
            input_ids,
            position_ids,
            max_new_tokens,
            stopping_criteria,
            position_ids_shift,
            decode=False
    ):
        if position_ids_shift.ndim == 1:
            position_ids_shift = position_ids_shift.unsqueeze(1)
        next_position_ids = position_ids[:, -1:] + position_ids_shift

        n_new_tokens = 0

        while n_new_tokens < max_new_tokens:
            outputs = self.base_model(input_ids=input_ids, position_ids=position_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            next_position_ids = next_position_ids + 1

            input_ids = torch.cat([input_ids, next_token_ids], dim=1)
            position_ids = torch.cat([position_ids, next_position_ids], dim=1)

            if stopping_criteria is not None:
                if stopping_criteria(input_ids, next_token_logits).all():
                    break
            elif (next_token_ids == self.tokenizer.eos_token_id).all():
                break

            n_new_tokens += 1
        
        if not decode:
            return input_ids
        
        texts_generated = self._decode_generated_ids(input_ids)
        return input_ids, texts_generated

    def _generate_with_new_tokens(
            self,
            input_ids,
            position_ids,
            max_new_tokens,
            stopping_criteria,
            new_tokens_start_id,
            decode=False
    ):
        # POSITION_IDS MAY WORK IMPROPERLY, NOT DEBUGGED

        device = input_ids.device
        batch_size = input_ids.shape[0]

        start_id_tensor = torch.tensor([new_tokens_start_id], dtype=torch.long).to(device)

        eos_count = torch.zeros(batch_size, dtype=torch.long).to(device)

        # When generating COT, append next_token + start_id
        # When generating answer, append next_token only
        # So we need to process two batches separately, updating corresponding masks each time
        cot_mask = torch.ones(batch_size, dtype=torch.bool).to(device)
        ans_mask = torch.zeros(batch_size, dtype=torch.bool).to(device)
        
        input_ids_list = [x.clone().detach() for x in input_ids]
        position_ids_list = [x.clone().detach() for x in position_ids] if position_ids else None

        # Prepend start_ids
        for i in range(batch_size):
            self._expand_with_next_tokens(input_ids_list, position_ids_list, [start_id_tensor], i=i)

        with torch.no_grad():
            for _ in range(0, max_new_tokens, 2):
                next_token_ids_list = [None for _ in range(batch_size)]
                self._generate_with_new_tokens_masked_step(input_ids_list, position_ids_list, next_token_ids_list, cot_mask)
                self._generate_with_new_tokens_masked_step(input_ids_list, position_ids_list, next_token_ids_list, ans_mask)
                
                for i, next_token_id in enumerate(next_token_ids_list):
                    if cot_mask[i]:
                        if next_token_id.item() == self.tokenizer.eos_token_id:
                            self._expand_with_next_tokens(input_ids_list, position_ids_list, [next_token_id], i)
                            cot_mask[i] = False
                            ans_mask[i] = True
                            eos_count[i] += 1
                        else:
                            self._expand_with_next_tokens(input_ids_list, position_ids_list, [next_token_id, start_id_tensor], i)
                    elif ans_mask[i]:
                        self._expand_with_next_tokens(input_ids_list, position_ids_list, [next_token_id], i)
                        if next_token_id.item() == self.tokenizer.eos_token_id:
                            ans_mask[i] = False
                            eos_count[i] += 1

                if stopping_criteria is not None:
                    if (eos_count >= 2).all():
                        break
                elif (eos_count >= 1).all():
                    break
        
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.eos_token_id)

        if not decode:
            return input_ids

        texts_generated = self._decode_generated_ids(input_ids)
        return input_ids, texts_generated

    def _generate_with_new_tokens_masked_step(self, input_ids_list, position_ids_list, next_token_ids_list, mask):
        if not mask.sum():
            return
        
        batch_size = len(input_ids_list)
        mask_indices = [i for i in range(batch_size) if mask[i]]

        first_length = len(input_ids_list[mask_indices[0]])
        if not all(len(input_ids_list[i]) == first_length for i in mask_indices):
            next_token_ids_masked = []
            for i in mask_indices:
                input_ids_masked = input_ids_list[i].unsqueeze(0)
                position_ids_masked = position_ids_list[i].unsqueeze(0) if position_ids_list is not None else None

                outputs_masked = self.base_model(input_ids=input_ids_masked, position_ids=position_ids_masked)
                next_token_logits_masked = outputs_masked.logits[:, -1, :]
                next_token_ids_masked.append(torch.argmax(next_token_logits_masked, dim=-1, keepdim=True))
        else:
            input_ids_masked = torch.cat([
                input_ids_list[i].unsqueeze(0)
                for i in mask_indices
            ], dim=0)
            
            position_ids_masked = torch.cat([
                position_ids_list[i].unsqueeze(0)
                for i in mask_indices
            ], dim=0) if position_ids_list is not None else None

            outputs_masked = self.base_model(input_ids=input_ids_masked, position_ids=position_ids_masked)
            next_token_logits_masked = outputs_masked.logits[:, -1, :]
            next_token_ids_masked = torch.argmax(next_token_logits_masked, dim=-1, keepdim=True)

        for i, next_token_id in zip(mask_indices, next_token_ids_masked):
            next_token_ids_list[i] = torch.tensor([next_token_id], dtype=torch.long).to(input_ids_list[i].device)
        
    def _expand_with_next_tokens(self, input_ids_list, position_ids_list, next_tokens_list, i):
        input_ids_list[i] = torch.cat([input_ids_list[i], *next_tokens_list], dim=0)

        if position_ids_list is not None:
            new_position_ids = torch.arange(
                input_ids_list[i].shape[0] - len(next_tokens_list), input_ids_list[i].shape[0]
            )
            position_ids_list[i] = torch.cat([position_ids_list[i], new_position_ids], dim=0)
    
    def _decode_generated_ids(self, input_ids):
        generated_sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, 2)
        texts_generated = []

        for i, input_ids_i in enumerate(input_ids):
            input_ids_i = input_ids_i[:generated_sep_positions[i]]
            texts_generated.append(self.tokenizer.decode(input_ids_i, skip_special_tokens=True))
        
        return texts_generated
        
    @classmethod
    def from_pretrained(self, pretrained_path, use_flash_attention=False):
        config = ImplicitModelConfig.from_pretrained(pretrained_path)
        model = ImplicitModel(config, reinitialize_weights=False, use_flash_attention=use_flash_attention)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'))
        model.load_state_dict(state_dict, strict=True)
        return model

    def save_pretrained(self, save_directory):
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))
