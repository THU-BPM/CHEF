import torch
import numpy as np



def replace_tokens(inputs, tokenizer, args):

    if not args.word_replacement > 0:
        return inputs

    new_inputs = inputs.clone()

    probability_matrix = torch.full(new_inputs.shape, args.word_replacement)

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in new_inputs.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = new_inputs.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    random_words = torch.randint(len(tokenizer), new_inputs.shape, dtype=torch.long)
    new_inputs[masked_indices] = random_words[masked_indices]

    return new_inputs

def freeze_full_bert(model, logger):
    # Freeze all bert pretrained weights
    for name, param in model.named_parameters():
        if 'classifier' not in name: # classifier layer
            logger.info('Freezing Parameters of Pretrained Model : ' + str(name))
            param.requires_grad = False
        else:
            logger.info('Not Freezing Parameters of Pretrained Model : ' + str(name))
