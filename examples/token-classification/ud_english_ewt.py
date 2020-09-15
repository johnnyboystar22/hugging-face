# coding=utf-8
# Copyright 2020 HuggingFace Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Universal Dependencies - English Dependency Treebank Universal Dependencies English Web Treebank v2.6 """

import logging

from conllu import parse_incr
from datasets import (
    BuilderConfig,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Sequence,
    Split,
    SplitGenerator,
    Value,
    Version,
)


_CITATION = """\
@inproceedings{silveira14gold,
  year = {2014},
  author = {Natalia Silveira and Timothy Dozat and Marie-Catherine de
  Marneffe and Samuel Bowman and Miriam Connor and John Bauer and
  Christopher D. Manning}, title = {A Gold Standard Dependency Corpus for {E}nglish},
  booktitle = {Proceedings of the Ninth International Conference on Language
  Resources and Evaluation (LREC-2014)}
}
"""

_DESCRIPTION = """\
The corpus comprises 254,830 words and 16,622 sentences, taken from five genres of web \n
media: weblogs, newsgroups, emails, reviews, and Yahoo! answers. See the LDC2012T13 documentation \n
for more details on the sources of the sentences. The trees were automatically converted into \n
Stanford Dependencies and then hand-corrected to Universal Dependencies. All the basic dependency \n
annotations have been single-annotated, a limited portion of them have been double-annotated, and \n
subsequent correction has been done to improve consistency. Other aspects of the treebank, such as \n
Universal POS, features and enhanced dependencies, has mainly been done automatically, with very limited hand-correction.
"""
_URL = "https://github.com/UniversalDependencies/UD_English-EWT/raw/master/"
_TRAINING_FILE = "en_ewt-ud-train.conllu"
_DEV_FILE = "en_ewt-ud-dev.conllu"
_TEST_FILE = "en_ewt-ud-test.conllu"


class UDEnglishEWTConfig(BuilderConfig):
    """BuilderConfig for UDEnglishEWT."""

    def __init__(self, **kwargs):
        """BuilderConfig for UDEnglishEWT.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(UDEnglishEWTConfig, self).__init__(**kwargs)


class UDEnglishEWT(GeneratorBasedBuilder):
    """Universal Dependencies - English Dependency Treebank Universal Dependencies English Web Treebank dataset."""

    BUILDER_CONFIGS = [
        UDEnglishEWTConfig(
            name="ud_english_ewt",
            version=Version("1.0.0"),
            description="Universal Dependencies - English Dependency Treebank Universal Dependencies English Web Treebank ",
        ),
    ]

    def _info(self):
        """See https://universaldependencies.org/format.html for CoNLL-U Format details"""
        return DatasetInfo(
            description=_DESCRIPTION,
            features=Features(
                {
                    "id": Value("string"),
                    "form": Sequence(Value("string")),
                    "upos": Sequence(Value("string")),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/UniversalDependencies/UD_English-EWT/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            SplitGenerator(name=Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logging.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            sentence_index = 0
            for sentence in parse_incr(f):
                form, upos = ([] for _ in range(2))
                for token in sentence:
                    form.append(token["form"])
                    upos.append(token["upos"])
                if form:
                    example = sentence_index, {
                        "id": str(sentence_index),
                        "form": form,
                        "upos": upos,
                    }
                    sentence_index += 1
                    yield example
