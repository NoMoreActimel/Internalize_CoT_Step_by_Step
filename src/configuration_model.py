from transformers import PretrainedConfig

class ImplicitModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        n_head=None,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = base_model
        super().__init__(**kwargs)

        if n_head is not None:
            assert hasattr(self, "n_head"), \
                f"Attempting to change n_head attribute which does not exist for the selected {base_model} model"
            assert self.n_embd % n_head == 0, \
                f"Incorrect n_head provided: {n_head}, n_embd = {self.n_embd} must be divisible by n_head"

            print(f"Changing the number of heads from {self.n_head} to {n_head}")
            self.n_head = n_head
