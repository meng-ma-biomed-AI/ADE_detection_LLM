import json
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from transformers import AutoModel, AutoConfig

from .transformers_chunking import apply_chunking_to_forward
from .modeling_utils import unwrap_model, get_parameter_dtype

"""Based on the hierarchical BERT implementation here: https://agit.ai/jsx/MCA_BERT/src/branch/master
"""

logger = logging.getLogger(__name__)


class OutputLayer(nn.Module):
    def __init__(self, Y, input_size):
        super(OutputLayer, self).__init__()
        self.Y = Y
        self.U = nn.Linear(input_size, Y)
        xavier_uniform_(self.U.weight)
        self.final = nn.Linear(input_size, Y)
        xavier_uniform_(self.final.weight)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x, target, text_inputs = None):
        att = self.U.weight.matmul(x.transpose(1, 2)) # [bs, Y, seq_len]
        alpha = F.softmax(att, dim=2)
        m = alpha.matmul(x) # [bs, Y, dim]
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        loss = self.loss_function(y, target)
        return loss, y


class ChunkBERT(nn.Module):
    """
    A BERT operating on documents chunked based on their length to chunks of length max 512.

    Forward returns document representation in format [batch size, number of features (768, hidden size), number of chunks (sequence length)]
    """

    def __init__(self, model_name_or_path,
                 cache_dir,
                 chunk_size=512):
        super(ChunkBERT, self).__init__()
        self.apply(self.init_bert_weights)
        self.chunk_size = chunk_size

        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.bert = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir, config=self.config)
    
    def forward_chunk(self, input_ids, token_type_ids, attention_mask):
        # x: [bs, seq_len]
        bert_out = self.bert(input_ids, token_type_ids, attention_mask,
                             return_dict=True)  # hidden_states: [bs, seq_len, 768]

        final_features = bert_out['last_hidden_state']

        # final_features = hidden_states[:, 0, :]   # the 0-th hidden state is used for classification
        return final_features

    def forward(self, input_ids, token_type_ids, attention_mask):
        # Document in format [batch size, number of features (768), number of chunks (sequence length)]
        document = apply_chunking_to_forward(self.forward_chunk,
                                             self.chunk_size,
                                             1,
                                             input_ids, token_type_ids, attention_mask)

        return document

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class HiBERT(nn.Module):

    def __init__(self, model_name_or_path, max_seq_len,
                 decoder,
                 num_labels,
                 cache_dir,
                 model='hibert',
                 n_heads=2,
                 transformer_encoder_num_layers=2,
                 chunk_size=512):
        super(HiBERT, self).__init__()

        self.MAX_LENGTH = max_seq_len
        self.model = model
        self.decoder = decoder
        self.apply(self.init_bert_weights)
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.bert = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir, config=self.config)

        self.chunk_bert = ChunkBERT(model_name_or_path, cache_dir, chunk_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.chunk_bert.bert.config.hidden_size,
                                                    nhead=n_heads,
                                                    dropout=self.chunk_bert.bert.config.hidden_dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=transformer_encoder_num_layers)

        if self.decoder == "fcn":
            self.classifier = nn.Sequential(nn.Dropout(p=self.chunk_bert.bert.config.hidden_dropout_prob), \
                                            nn.Linear(self.chunk_bert.bert.config.hidden_size, num_labels))
        elif self.decoder == "lan":
            self.output_layer = OutputLayer(num_labels, self.chunk_bert.bert.config.hidden_size)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        # x: [bs, seq_len]
        document = self.chunk_bert(input_ids, token_type_ids, attention_mask)
        document = document.permute(2,0,1)
        transformer_output = self.transformer_encoder(document)
        transformer_output = transformer_output.permute(1,0,2)
        if self.decoder == 'fcn':
            y = self.classifier(transformer_output.max(dim=1)[0])
            loss = self.loss_fn(y, labels)
        elif self.decoder == 'lan':
            loss, y = self.output_layer(transformer_output, labels)
        return loss, y

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def save_pretrained(
            self,
            save_directory,
            fname='pytorch_model.bin',
            save_config = True,
            state_dict = None,
            save_function = torch.save,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~PreTrainedModel.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            fname (str): Output model file name. Defaults to 'pytorch_model.bin'
            save_config (`bool`, *optional*, defaults to `True`):
                Whether or not to save the config of the model. Useful when in distributed training like TPUs and need
                to call this function on all processes. In this case, set `save_config=True` only on the main process
                to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # dtype = get_parameter_dtype(model_to_save)
        # model_to_save.bert.config.torch_dtype = str(dtype).split(".")[1]

        # Save the config
        if save_config:
            model_to_save.config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        # if self._keys_to_ignore_on_save is not None:
        #     for ignore_key in self._keys_to_ignore_on_save:
        #         if ignore_key in state_dict.keys():
        #             del state_dict[ignore_key]

        # If we save using the predefined names, we can load using ``from_pretra`ined`
        output_model_file = os.path.join(save_directory, fname)
        save_function(state_dict, output_model_file)

        logger.info(f"Model weights saved in {output_model_file}")

    @classmethod
    def from_pretrained(cls, bert_path, max_seq_len, decoder, num_labels, cache_dir,
                        model_dir, model_name='pytorch_model.bin'):

        model = cls(bert_path, max_seq_len, decoder, num_labels, cache_dir)

        model_file = os.path.join(model_dir, model_name)

        state_dict = torch.load(model_file, map_location="cpu")
        # torch_dtype = next(iter(state_dict.values())).dtype
        # dtype_orig = cls._set_default_torch_dtype(torch_dtype)
        # torch.set_default_dtype(dtype_orig)

        model.load_state_dict(state_dict)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        return model

