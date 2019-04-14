from typing import Iterator, List, Dict

import torch
import torch.nn as nn
import shutil
import numpy as np
import tempfile
import glob
import unicodedata
import string

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.params import Params
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, Entropy
from allennlp.commands.train import train_model
from allennlp.predictors import SentenceTaggerPredictor


@Model.register('lstm-tagger')
class RNN(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))

        self.loss = nn.CrossEntropyLoss()

    def forward(self,
                word: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(word)
        embeddings = self.word_embeddings(word)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        # I just need 18 values for each output (1 for each class)
        tag_logits = tag_logits[:, -1]
        output = {"tag_logits": tag_logits}
        if label is not None:
            loss = self.loss(tag_logits, label.squeeze(-1))
            output["loss"] = loss

        return output



@DatasetReader.register('pos-tutorial')
class PosDatasetReader(DatasetReader):

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: [Token], tag: List[str] = None) -> Instance:
        word_field = TextField(tokens, self.token_indexers)
        fields = {"word": word_field}

        if tag:
            label_field = LabelField(tag)
            fields["label"] = label_field
        return Instance(fields)

    def readLines(self, filename):
        lines = open(filename).read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]

    def unicodeToAscii(self, s):
        all_letters = string.ascii_letters + " .,;'-"
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )

    def _read(self, file_path: str) -> Iterator[Instance]:
        for filename in glob.glob(file_path + '/*.txt'):
            label = filename.split("/")[-1].split(".")[0]
            names = self.readLines(filename)
            for name in names:
                a = self.text_to_instance([Token(letter) for letter in name], label)
                yield self.text_to_instance([Token(letter) for letter in name], label)


if __name__ == "__main__":
    params = Params.from_file('hw4.jsonnet')
    serialization_dir = tempfile.mkdtemp()
    model = train_model(params, serialization_dir)

    # Make predictions
    predictor = SentenceTaggerPredictor(model, dataset_reader=PosDatasetReader())
    tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
    tag_ids = np.argmax(tag_logits, axis=-1)

    shutil.rmtree(serialization_dir)
