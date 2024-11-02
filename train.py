import torch, copy
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead, RobertaEmbeddings, RobertaEncoder, RobertaPooler
)
from transformers import (
    RobertaPreTrainedModel,
    RobertaModel,
    DataCollatorForLanguageModeling,
    Trainer,
    PreTrainedModel,
    TrainingArguments,
    RobertaConfig
)
from transformers.modeling_outputs import MaskedLMOutput

class MAELM_PretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RobertaEmbeddings", "RobertaSelfAttention"]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class MAELM(MAELM_PretrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
        

class MAELM_for_MLM(MAELM_PretrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.config=config
        
        # Encoder: BERT model with the target size
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        
        self.aux_config = copy.deepcopy(config)
        self.aux_config.num_hidden_layers = 4
        self.aux_config.hidden_dropout_prob = 0
        self.aux_config.attention_probs_dropout_prob = 0
        
        self.aux_encoder = RobertaEncoder(self.aux_config)        
        
        # Initialize weights
        self.post_init()
        
    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
    
    def forward(self, input_ids, 
                attention_mask=None, token_type_ids=None, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        masked_indices = labels != -100
        mask_token_embedding = self.roberta.embeddings.word_embeddings.weight[self.config.mask_token_id]
        
        # Encoder input: mask out the masked positions
        encoder_input_ids = input_ids.masked_fill(masked_indices, self.config.pad_token_id)
        encoder_attention_mask = (encoder_input_ids != self.config.pad_token_id).long()
        
        # Encoder: process unmasked tokens
        encoder_outputs = self.roberta(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        mask_positions = masked_indices.nonzero(as_tuple=False)  # Shape: [num_masks, 2] (batch_idx, seq_idx)
        mask_token_embedding = mask_token_embedding.unsqueeze(0).expand(mask_positions.size(0), -1)  # [num_masks, hidden_size]
        
        # Retrieve the positional embeddings for the masked positions        
        batch_indices = mask_positions[:, 0]
        seq_indices = mask_positions[:, 1]
        position_ids = seq_indices
        position_embeddings = self.roberta.embeddings.position_embeddings(position_ids)  # [num_masks, hidden_size]
        
        if token_type_ids is not None:
            token_type = token_type_ids[batch_indices, seq_indices]  # [num_masks]
            token_type_embeds = self.roberta.embeddings.token_type_embeddings(token_type)  # [num_masks, hidden_size]
        else:
            token_type_embeds = 0  # If no token type embeddings
        
        # Combine Mask Token Embedding with Positional and Token Type Embeddings
        enriched_mask_embeddings = mask_token_embedding + position_embeddings + token_type_embeds  # [num_masks, hidden_size]
        hidden_states_w_mask = encoder_hidden_states.clone()
        
        # Replace the embeddings at masked positions with enriched mask embeddings
        hidden_states_w_mask[batch_indices, seq_indices] = enriched_mask_embeddings

        # Prepare attention mask for aux_encoder
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_attention_mask = attention_mask[:, None, None, :]
        
        # aux_enc: process masked tokens with cross-attention to encoder outputs
        aux_outputs = self.aux_encoder(
            hidden_states=hidden_states_w_mask,
            attention_mask=extended_attention_mask, # shape = (bsz,1,1,len)
            return_dict=return_dict
        )
        aux_hidden_states = aux_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        prediction_scores = self.lm_head(aux_hidden_states)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + aux_outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=aux_outputs.hidden_states,
            attentions=aux_outputs.attentions,
        )

        
if __name__ == '__main__':
    from transformers import AutoTokenizer, RobertaConfig, DataCollatorForLanguageModeling, 
    from transformers import Trainer, TrainingArguments
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", model_max_length=128, is_fast=False)
    tokenizer.return_special_tokens_mask=True

    encoder_config = RobertaConfig()
    model = MAELM_for_MLM(encoder_config)
    model.init_weights()
    model.resize_token_embeddings(new_num_tokens=tokenizer.vocab_size)

    dataset = load_dataset("lhoestq/demo1")

    def tokenize(row):
        return tokenizer(row['review'], padding='max_length', max_length=128, truncation=True)
    tokenized_toy_data = toy_data.map(tokenize, remove_columns='label')
    tokenized_toy_data = tokenized_toy_data.train_test_split(test_size=0.1)

    # Data collator for MLM
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='maelm_test_toy',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
        per_device_train_batch_size=2,
        # gradient_accumulation_steps=8,
        # dataloader_num_workers=2,
        # tf32=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_toy_data['train'],
        eval_dataset=tokenized_toy_data['test'],
        data_collator=collator,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()
