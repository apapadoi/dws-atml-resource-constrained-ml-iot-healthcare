from chronos import ChronosPipeline, ChronosModel, ChronosConfig
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GPTQConfig
)
import torch

class GPTQChronosPipeline(ChronosPipeline):
    @classmethod
    def from_pretrained(cls, model_id, output_dir, chronos_config, *args, **kwargs):
        args = args + (model_id,)

        config = AutoConfig.from_pretrained(*args, **kwargs)
        config.chronos_config = chronos_config

        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        chronos_config = ChronosConfig(**config.chronos_config)

        tokenizer=chronos_config.create_tokenizer()

        if chronos_config.model_type == "seq2seq":
            inner_model = AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
        else:
            assert chronos_config.model_type == "causal"
            
            inner_model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)

            tokenizer_len = chronos_config.n_tokens - chronos_config.n_special_tokens - 1

            inner_model.resize_token_embeddings(tokenizer_len)

            old_embedding_weights = inner_model.get_input_embeddings().weight.data
            new_embedding_weights = old_embedding_weights[:tokenizer_len, :]
            inner_model.get_input_embeddings().weight.data = new_embedding_weights

        inner_model.save_pretrained(output_dir)
        
        return cls(
            tokenizer=tokenizer,
            model=ChronosModel(config=chronos_config, model=inner_model),
        )
