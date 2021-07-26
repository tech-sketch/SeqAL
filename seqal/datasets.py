import re
from pathlib import Path
from typing import Dict, List, Union

from flair.data import Corpus as ParentCorpus
from flair.data import FlairDataset, Sentence, Token
from flair.datasets.base import find_train_dev_test_files
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset


class Corpus(ParentCorpus):
    """The modified Corpus class.

    Args:
        ParentCorpus ([type]): [description]
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

    def remove_query_samples(self, query_idx: List[int]) -> None:
        """Remove queried data from data pool.

        Args:
            query_idx (List[int]): Index list of queried data.
        """
        self.test.sentences = [
            sent for i, sent in enumerate(self.test.sentences) if i not in query_idx
        ]

    def add_query_samples(self, query_samples: List[Sentence]) -> None:
        """Add queried data to labeled data.

        Args:
            query_idx (int): Index list of queried data.
        """
        for sample in query_samples:
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
                banned_sentences=banned_sentences,
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
                banned_sentences=banned_sentences,
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
                banned_sentences=banned_sentences,
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


class ColumnDataset(FlairDataset):
    # special key for space after
    SPACE_AFTER_KEY = "space-after"

    def __init__(
        self,
        path_to_column_file: Union[str, Path],
        column_name_map: Dict[int, str],
        tag_to_bioes: str = None,
        column_delimiter: str = r"\s+",
        comment_symbol: str = None,
        banned_sentences: List[str] = None,
        in_memory: bool = True,
        document_separator_token: str = None,
        encoding: str = "utf-8",
        skip_first_line: bool = False,
        label_name_map: Dict[str, str] = None,
    ):
        """Instantiates a column dataset (typically used for sequence labeling or word-level prediction).

        Args:
            path_to_column_file (Union[str, Path]): Path to the file with the column-formatted data
            column_name_map (Dict[int, str]): A map specifying the column format
            tag_to_bioes (str, optional): Whether to convert to BIOES tagging scheme
            column_delimiter (str, optional): Default is to split on any separatator,
                                              but you can overwrite for instance with "\t"
            comment_symbol (str, optional): If set, lines that begin with this symbol are treated as comments
            banned_sentences (List[str], optional): If set to True, the dataset is kept in memory as Sentence objects,
                                                    otherwise does disk reads
            in_memory (bool, optional): If set to True, the dataset is kept in memory as Sentence objects,
                                        otherwise does disk reads
            document_separator_token (str, optional): If provided, sentences that function as document
                                                      boundaries are so marked
            encoding (str, optional): Encodings. Defaults to "utf-8".
            skip_first_line (bool, optional): Optionally map tag names to different schema.
            label_name_map (Dict[str, str], optional): Optionally map tag names to different schema.

        Returns:
            Dataset: a dataset with annotated data
        """
        if type(path_to_column_file) is str:
            path_to_column_file = Path(path_to_column_file)
        assert path_to_column_file.exists()
        self.path_to_column_file = path_to_column_file
        self.tag_to_bioes = tag_to_bioes
        self.column_name_map = column_name_map
        self.column_delimiter = column_delimiter
        self.comment_symbol = comment_symbol
        self.document_separator_token = document_separator_token
        self.label_name_map = label_name_map
        self.banned_sentences = banned_sentences

        # store either Sentence objects in memory, or only file offsets
        self.in_memory = in_memory

        self.total_sentence_count: int = 0

        # most data sets have the token text in the first column, if not, pass 'text' as column
        self.text_column: int = 0
        for column in self.column_name_map:
            if column_name_map[column] == "text":
                self.text_column = column

        # determine encoding of text file
        self.encoding = encoding

        with open(str(self.path_to_column_file), encoding=self.encoding) as file:

            # skip first line if to selected
            if skip_first_line:
                file.readline()

            # option 1: read only sentence boundaries as offset positions
            if not self.in_memory:
                self.indices: List[int] = []

                line = file.readline()
                position = 0
                sentence_started = False
                while line:
                    if sentence_started and self.__line_completes_sentence(line):
                        self.indices.append(position)
                        position = file.tell()
                        sentence_started = False

                    elif not line.isspace():
                        sentence_started = True
                    line = file.readline()

                if sentence_started:
                    self.indices.append(position)

                self.total_sentence_count = len(self.indices)

            # option 2: keep everything in memory
            if self.in_memory:
                self.sentences: List[Sentence] = []

                # pointer to previous
                previous_sentence = None
                while True:
                    sentence = self._convert_lines_to_sentence(
                        self._read_next_sentence(file)
                    )
                    if not sentence:
                        break
                    if self.banned_sentences is not None and any(
                        [d in sentence.to_plain_string() for d in self.banned_sentences]
                    ):
                        continue
                    sentence._previous_sentence = previous_sentence
                    sentence._next_sentence = None

                    if previous_sentence:
                        previous_sentence._next_sentence = sentence

                    self.sentences.append(sentence)
                    previous_sentence = sentence

                self.total_sentence_count = len(self.sentences)

    def _read_next_sentence(self, file):
        lines = []
        line = file.readline()
        while line:
            if not line.isspace():
                lines.append(line)

            # if sentence ends, break
            if len(lines) > 0 and self.__line_completes_sentence(line):
                break

            line = file.readline()
        return lines

    def _convert_lines_to_sentence(self, lines):

        sentence: Sentence = Sentence()
        for line in lines:
            # skip comments
            if self.comment_symbol is not None and line.startswith(self.comment_symbol):
                continue

            # if sentence ends, convert and return
            if self.__line_completes_sentence(line):
                if len(sentence) > 0:
                    if self.tag_to_bioes is not None:
                        sentence.convert_tag_scheme(
                            tag_type=self.tag_to_bioes, target_scheme="iobes"
                        )
                    # check if this sentence is a document boundary
                    if sentence.to_original_text() == self.document_separator_token:
                        sentence.is_document_boundary = True
                    return sentence

            # otherwise, this line is a token. parse and add to sentence
            else:
                token = self._parse_token(line)
                sentence.add_token(token)

        # check if this sentence is a document boundary
        if sentence.to_original_text() == self.document_separator_token:
            sentence.is_document_boundary = True

        if self.tag_to_bioes is not None:
            sentence.convert_tag_scheme(
                tag_type=self.tag_to_bioes, target_scheme="iobes"
            )

        if len(sentence) > 0:
            return sentence

    def _parse_token(self, line: str) -> Token:
        fields: List[str] = re.split(self.column_delimiter, line.rstrip())
        token = Token(fields[self.text_column])
        for column in self.column_name_map:
            if len(fields) > column:
                if (
                    column != self.text_column
                    and self.column_name_map[column] != self.SPACE_AFTER_KEY
                ):
                    task = self.column_name_map[column]  # for example 'pos'
                    tag = fields[column]
                    if tag.count("-") >= 1:  # tag with prefix, for example tag='B-OBJ'
                        split_at_first_hyphen = tag.split("-", 1)
                        tagging_format_prefix = split_at_first_hyphen[0]
                        tag_without_tagging_format = split_at_first_hyphen[1]
                        if (
                            self.label_name_map
                            and tag_without_tagging_format in self.label_name_map.keys()
                        ):
                            tag = (
                                tagging_format_prefix
                                + "-"
                                + self.label_name_map[tag_without_tagging_format]
                            )
                            # for example, transforming 'B-OBJ' to 'B-part-of-speech-object'
                            if self.label_name_map[tag_without_tagging_format] == "O":
                                tag = "O"
                    else:  # tag without prefix, for example tag='PPER'
                        if self.label_name_map and tag in self.label_name_map.keys():
                            tag = self.label_name_map[
                                tag
                            ]  # for example, transforming 'PPER' to 'person'

                    token.add_label(task, tag)
                if (
                    self.column_name_map[column] == self.SPACE_AFTER_KEY
                    and fields[column] == "-"
                ):
                    token.whitespace_after = False
        return token

    def __line_completes_sentence(self, line: str) -> bool:
        sentence_completed = line.isspace() or line == ""
        return sentence_completed

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int = 0) -> Sentence:

        # if in memory, retrieve parsed sentence
        if self.in_memory:
            sentence = self.sentences[index]

        # else skip to position in file where sentence begins
        else:
            with open(str(self.path_to_column_file), encoding=self.encoding) as file:
                file.seek(self.indices[index])
                sentence = self._convert_lines_to_sentence(
                    self._read_next_sentence(file)
                )

            # set sentence context using partials
            sentence._position_in_dataset = (self, index)

        return sentence
