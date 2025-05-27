from transformers import AutoConfig, AutoModelForCausalLM

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
