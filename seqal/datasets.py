from pathlib import Path
from typing import Dict, List, Union

from flair.data import Corpus as ParentCorpus
from flair.data import Sentence
from flair.datasets import ColumnDataset as ParentColumnDataset
from flair.datasets.base import find_train_dev_test_files
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset


class Corpus(ParentCorpus):
    """The modified Corpus class.

    Args:
        ParentCorpus: The original Corpus class.
    """

    def get_all_sentences(self) -> Dataset:
        """Refactor method of flair.data.corpus

        Returns:
            Dataset: flair dataset.
        """
        parts = []
        if self.train:
            parts.append(self.train.sentences)
        if self.dev:
            parts.append(self.dev.sentences)
        if self.test:
            parts.append(self.test.sentences)
        return ConcatDataset(parts)

    def add_queried_samples(self, queried_samples: List[Sentence]) -> None:
        """Add queried data to labeled data.

        Args:
            queried_samples (List[Sentence]): Queried data.
        """
        for sample in queried_samples:
            self.train.sentences.append(sample)


class ColumnCorpus(Corpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        column_format: Dict[int, str],
        train_file=None,
        test_file=None,
        dev_file=None,
        tag_to_bioes=None,
        column_delimiter: str = r"\s+",
        comment_symbol: str = None,
        encoding: str = "utf-8",
        document_separator_token: str = None,
        skip_first_line: bool = False,
        in_memory: bool = True,
        label_name_map: Dict[str, str] = None,
        banned_sentences: List[str] = None,
        autofind_splits: bool = True,
        **corpusargs,
    ):
        """Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        Args:
            data_folder (Union[str, Path]): Base folder with the task data
            column_format (Dict[int, str]): A map specifying the column format
            train_file ([type], optional): The name of the train file
            test_file ([type], optional): The name of the test file
            dev_file ([type], optional): The name of the dev file, if None, dev data is sampled from train
            tag_to_bioes ([type], optional): Whether to convert to BIOES tagging scheme
            column_delimiter (str, optional): Default is to split on any separatator,
                                              but you can overwrite for instance with "\t" to split only on tabs
            comment_symbol (str, optional): If set, lines that begin with this symbol are treated as comments
            encoding (str, optional): Encodings. Defaults to "utf-8".
            document_separator_token (str, optional): If provided, sentences that function as document
                                                      boundaries are so marked
            skip_first_line (bool, optional): Set to True if your dataset has a header line
            in_memory (bool, optional): If set to True, the dataset is kept in memory as Sentence objects,
                                        otherwise does disk reads
            label_name_map (Dict[str, str], optional): Optionally map tag names to different schema.
            banned_sentences (List[str], optional): Optionally remove sentences from the corpus.
                                                    Works only if `in_memory` is true
            autofind_splits (bool, optional): Defaults to True.

        Returns:
            Dataset: a Corpus with annotated train, dev and test data
        """

        # find train, dev and test files if not specified
        dev_file, test_file, train_file = find_train_dev_test_files(
            data_folder, dev_file, test_file, train_file, autofind_splits
        )

        # get train data
        train = (
            ColumnDataset(
                train_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                column_delimiter=column_delimiter,
                # banned_sentences=banned_sentences,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
                skip_first_line=skip_first_line,
                label_name_map=label_name_map,
            )
            if train_file is not None
            else None
        )

        # read in test file if exists
        test = (
            ColumnDataset(
                test_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                column_delimiter=column_delimiter,
                # banned_sentences=banned_sentences,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
                skip_first_line=skip_first_line,
                label_name_map=label_name_map,
            )
            if test_file is not None
            else None
        )

        # read in dev file if exists
        dev = (
            ColumnDataset(
                dev_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                # banned_sentences=banned_sentences,
                column_delimiter=column_delimiter,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
                skip_first_line=skip_first_line,
                label_name_map=label_name_map,
            )
            if dev_file is not None
            else None
        )

        super(ColumnCorpus, self).__init__(
            train, dev, test, name=str(data_folder), **corpusargs
        )


class ColumnDataset(ParentColumnDataset):
    def __len__(self):
        """Override method"""
        return len(self.sentences)

    def obtain_statistics(self, name: str = "Pool", tag_type: str = None):
        return Corpus._obtain_statistics_for(
            self.sentences, name=name, tag_type=tag_type
        )
