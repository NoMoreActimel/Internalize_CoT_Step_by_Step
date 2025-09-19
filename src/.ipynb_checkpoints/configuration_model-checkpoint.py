from transformers import PretrainedConfig
from transformers import AutoConfig, AutoModelForCausalLM


# class ImplicitModelConfig(PretrainedConfig):
#     def __init__(
#         self,
#         base_model='gpt2',
#         n_head=None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)

#         defaults = AutoConfig.from_pretrained(base_model).to_dict()
#         defaults.update(kwargs)
#         super().__init__(**defaults)

#         self.base_model = base_model
#         self.tokenizer_name = base_model

#         print("N_HEAD:", n_head)

#         if n_head is not None:
#             emb = getattr(self, "n_embd", None) or getattr(self, "d_model", None)

#             # assert hasattr(self, "n_head"), \
#             #     f"Attempting to change n_head attribute which does not exist for the selected {base_model} model"
#             if emb is not None:
#                 assert self.n_embd % n_head == 0, \
#                     f"Incorrect n_head provided: {n_head}, n_embd = {self.n_embd} must be divisible by n_head"

#             if hasattr(self, "num_attention_heads"):
#                 print(f"Layer {idx + 1}, overriding num_attention_heads: {self.num_attention_heads} to {n_head}")

#             self.num_attention_heads = n_head
            
#             # for idx, layer in enumerate(model.base_model.transformer.h):
#             #     if hasattr(layer, "attn"):
#             #         if hasattr(self, "n_head"):
#             #             print(f"Layer {idx + 1}, overriding num_heads: {layer.attn.num_heads} to {n_head}")


class ImplicitModelConfig:
    def __init__(self, base_model="gpt2", n_head=None, **kwargs):
        self.config = AutoConfig.from_pretrained(base_model)
        
        self.base_model = base_model
        self.tokenizer_name = base_model

        if n_head is not None:
            if not hasattr(self.config, "num_attention_heads"):
                print("Attempting to change num_attention_heads, while this attribute does not exist in config!")
            else:
                print("Old num_attention_heads:", self.config.num_attention_heads)
            
            print("New num_attention_heads:", n_head)
            self.config.num_attention_heads = n_head

    @classmethod
    def from_pretrained(cls, base_model="gpt2", n_head=None, **kwargs):
        cfg = AutoConfig.from_pretrained(base_model, **kwargs)

        if n_head is not None:
            emb = getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None)
            if emb is not None and emb % n_head != 0:
                raise ValueError(f"{emb} not divisible by {n_head}")

            if not hasattr(cfg, "num_attention_heads"):
                print("Attempting to change num_attention_heads, while this attribute does not exist in config!")
            else:
                print(f"Overriding num_attention_heads: {cfg.num_attention_heads} to {n_head}")
            
            cfg.num_attention_heads = n_head

        return cls(cfg, base_model)

    def save_pretrained(self, *args, **kwargs):
        return self.config.save_pretrained(*args, **kwargs)

    def create_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.base_model,
            config=self.config
        )
