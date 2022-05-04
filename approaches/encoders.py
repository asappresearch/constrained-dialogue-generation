from transformers import GPT2LMHeadModel


class GPT2Encoder(GPT2LMHeadModel):
    def __init__(self, config):
        super(GPT2Encoder, self).__init__(config)

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        transformer_outputs = self.transformer(input_ids,
                                               past_key_values=past_key_values,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds,
                                               encoder_hidden_states=encoder_hidden_states,
                                               encoder_attention_mask=encoder_attention_mask,
                                               use_cache=use_cache,
                                               output_attentions=output_attentions,
                                               output_hidden_states=output_hidden_states,
                                               return_dict=return_dict)
        hidden_states = transformer_outputs.last_hidden_state[:, -1, :]

        return hidden_states
