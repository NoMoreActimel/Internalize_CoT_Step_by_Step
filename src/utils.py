import os
import torch
from transformers import StoppingCriteria, LogitsProcessor


COT_ANSWER_SPLIT_PATTERN = "####"

def save_model(model, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def batch_ids(input_ids_list, pad_token_id, device, dtype):
    max_seq_len = max([len(item) for item in input_ids_list])
    batch_size = len(input_ids_list)
    input_ids = torch.Tensor(batch_size, max_seq_len).to(dtype).to(device)
    input_ids.fill_(pad_token_id)
    for batch_id in range(batch_size):
        input_ids[batch_id, :len(input_ids_list[batch_id])] = input_ids_list[batch_id]
    return input_ids


def get_sep_position(input_ids, sep_id, skip=0):
    def get_next_nonzero(mask, default_value=None):
        mask_nonzero = mask.nonzero()
        if mask_nonzero.shape[0]:
            return mask_nonzero[0, -1].item()
        print("Next nonzero element not found in get_sep_position!")
        return default_value
    
    batch_size = input_ids.shape[0]
    sep_positions = input_ids.new_zeros(batch_size).long()

    for batch_id in range(batch_size):
        mask = input_ids[batch_id].eq(sep_id)
        sep_position = get_next_nonzero(mask, input_ids.shape[1])

        if sep_position < input_ids.shape[1]:
            for _ in range(skip):
                mask[sep_position] = False
                sep_position = get_next_nonzero(mask, input_ids.shape[1])
                if sep_position == input_ids.shape[1]:
                    break
                
        sep_positions[batch_id] = sep_position

    return sep_positions


def extract_answer(text):
    split_pattern = COT_ANSWER_SPLIT_PATTERN
    if split_pattern not in text:
        return text.strip().replace(',', '')
    else:
        _, ans = text.strip().split(COT_ANSWER_SPLIT_PATTERN, 1)
        ans = COT_ANSWER_SPLIT_PATTERN + ans
        ans = ans.strip().replace(',', '')
        return ans

def extract_cot(text):
    split_pattern = COT_ANSWER_SPLIT_PATTERN
    if split_pattern not in text:
        #import pdb; pdb.set_trace()
        return None
    else:
        cot, _ = text.strip().split(COT_ANSWER_SPLIT_PATTERN, 1)
        cot = cot.strip()
        return cot


# Stop generation only after generating two EOSs, such as  z <eos> y <eos>
class DoubleEOSStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.init = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        eos_count = (input_ids == self.eos_token_id).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        return done.all()

class DoubleEOSLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.init = False
    
    def __call__(self, input_ids, scores):
        eos_count = (input_ids == self.eos_token_id).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        if done.any():
            scores[done, :] = float('-inf')
            scores[done, self.eos_token_id] = 0
        return scores
