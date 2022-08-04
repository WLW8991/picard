import os
import sys
abs_path = os.path.abspath(sys.argv[0])
dir_path = abs_path.replace("/seq2seq/datasets/lc_quad_2/lc_quad.py", "")
sys.path.append(dir_path)


import json
import re
import datasets
from seq2seq.datasets.lc_quad_2.preprocess import Preprocess


# TODO(lc_quad): BibTeX citation
_CITATION = """
@inproceedings{dubey2017lc2,
title={LC-QuAD 2.0: A Large Dataset for Complex Question Answering over Wikidata and DBpedia},
author={Dubey, Mohnish and Banerjee, Debayan and Abdelkawi, Abdelrahman and Lehmann, Jens},
booktitle={Proceedings of the 18th International Semantic Web Conference (ISWC)},
year={2019},
organization={Springer}
}
"""

# TODO(lc_quad):
_DESCRIPTION = """\
LC-QuAD 2.0 is a Large Question Answering dataset with 30,000 pairs of question and its corresponding SPARQL query. The target knowledge base is Wikidata and DBpedia, specifically the 2018 version. Please see our paper for details about the dataset creation process and framework.
"""
_URL = "https://github.com/AskNowQA/LC-QuAD2.0/archive/master.zip"


class LcQuad(datasets.GeneratorBasedBuilder):
    """TODO(lc_quad): Short description of my dataset."""

    # TODO(lc_quad): Set up version.
    VERSION = datasets.Version("2.0.0")

    def _info(self):
        # TODO(lc_quad): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "NNQT_question": datasets.Value("string"),
                    "uid": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "sparql_wikidata": datasets.Value("string"),
                    "question_process": datasets.Value("string"),
                    "sparql_process": datasets.Value("string"),
                    "sparql_dbpedia18": datasets.Value("string"),
                    #"paraphrased_question": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="http://lc-quad.sda.tech/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(lc_quad): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        # dl_dir = './transform/transformers_cache/downloads'
        dl_dir = os.path.join(dl_dir, "LC-QuAD2.0-master", "dataset")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(dl_dir, "train.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(dl_dir, "test.json")},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(lc_quad): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            pre = Preprocess()

            for id_, row in enumerate(data):
                #is_list = False
                #for key in row:
                #    if key != "answer" and isinstance(row[key], list):
                #        is_list = True
                #if is_list:
                 #   continue
                gold_query, question_input = pre._preprocess(row)
                yield id_, {
                    "NNQT_question": row["NNQT_question"],
                    "uid": row["uid"],
                    "question": row["question"],
                    "sparql_wikidata": row["sparql_wikidata"],
                    "question_process": question_input,
                    "sparql_process": gold_query,
                    "sparql_dbpedia18": row["sparql_dbpedia18"],
                    #"paraphrased_question": row["paraphrased_question"],
                }