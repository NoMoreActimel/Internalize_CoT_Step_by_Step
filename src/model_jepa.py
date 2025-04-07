import os
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss, MSELoss
from model import ImplicitModel

from utils import get_sep_position

class JEPAImplicitModel(ImplicitModel):
    def __init__(
            self,
            config,
            ref_model,
            alpha_logits_loss=1.0,
            ref_model_update_decay=0.999,
            reinitialize_weights=False,
            use_flash_attention=False
    ):
        """
            The model is assumed to be trained on partially corrupted COT data,
            recovering the corresponding answer logits of full-COT-pretrained reference model.

            That is, reference model is pretrained on original COT data: [query] [........COT........] [answer]
            While this model is trained on partially masked COT data:    [query] [M..MM...MMM...MMMMM] [answer]
        """
        super().__init__(config, reinitialize_weights, use_flash_attention)
        self.alpha_logits_loss = alpha_logits_loss
        self.ref_model_update_decay = ref_model_update_decay
        self.ref_model = ref_model
        self.ref_model.eval()

    def compute_loss(self, input_ids, labels, position_ids=None, output_attentions=False):
        outputs = self.forward(input_ids=input_ids, position_ids=position_ids, output_attentions=output_attentions)
        logits = outputs.logits

        ref_outputs = self.ref_model.forward(input_ids=input_ids, position_ids=position_ids, output_attentions=output_attentions)
        ref_logits = ref_outputs.logits

        answer_pos = get_sep_position(labels, sep_id=self.tokenizer.eos_token_id, skip=1) + 1
        answer_logits = logits[..., answer_pos:]
        ref_answer_logits = ref_logits[..., answer_pos:]
        logits_loss = nn.functional.mse_loss(answer_logits, ref_answer_logits)

        labels_pred = logits.argmax(-1)
        mask = labels[...,1:].ge(0)
        correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        outputs.ce_loss = loss
        outputs.logits_loss = logits_loss
        outputs.loss = loss + self.alpha_logits_loss * logits_loss

        outputs.token_accuracy = token_accuracy
        outputs.total_correct = correct_tokens
        outputs.total_loss = loss * total_tokens
        outputs.total_tokens = total_tokens

        return outputs
    
    def update_ref_model(self, decay=None):
        # EMA on self.ref_model weights
        if decay is None:
            decay = self.ref_model_update_decay
        with torch.no_grad():
            for ref_param, student_param in zip(self.ref_model.base_model.parameters(), self.base_model.parameters()):
                ref_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)
