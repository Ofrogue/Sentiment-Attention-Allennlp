import torch
from allennlp.nn.util import get_final_encoder_states
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder


@Seq2VecEncoder.register("attention_encoder")
class AttentionSeq2Veq(Seq2VecEncoder):
	def __init__(self,
				input_dim: int,
                hidden_dim: int,
                projection_dim: int,
                feedforward_hidden_dim: int,
                num_layers: int,
				num_attention_heads: int):

		super(AttentionSeq2Veq, self).__init__(stateful=False)

		self._seq2seq = StackedSelfAttentionEncoder(input_dim=input_dim,
													hidden_dim=hidden_dim,
													projection_dim=projection_dim,
													feedforward_hidden_dim=feedforward_hidden_dim,
													num_layers=num_layers,
													num_attention_heads=num_attention_heads)

		self._hidden_dim = hidden_dim
		self._input_dim = input_dim


	def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
		# https://github.com/allenai/allennlp/issues/2411
		return get_final_encoder_states(
										self._seq2seq(inputs, None),
										mask
										)


	def get_output_dim(self):
		return self._hidden_dim

	def get_input_dim(self):
		return self._input_dim
