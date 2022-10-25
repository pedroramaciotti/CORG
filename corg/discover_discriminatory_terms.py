"""accepts a csv file of the form text|dimension_1|dimension_2|dimension_n
   and returns a scsv file of the form lemma|perplexity
"""

import os

import pandas as pd
pd.set_option("display.max_columns", None)

import spacy
from spacy.tokenizer import _get_regex_pattern

import re

import textacy

class DiscriminatoryTermsExtractor:

    def load_text_and_dimensions(self, text_and_dimensions_filename = None):
        if text_and_dimensions_filename is None:
            raise ValueError('Text and dimensions filename should be provided.')

        if not os.path.isfile(text_and_dimensions_filename):
            raise ValueError('Text and dimensions filename does not exist.')

        df = pd.read_csv(text_and_dimensions_filename, on_bad_lines = 'skip')

        return (df)

    # if sample_no not None use a sample of the documents to create the corpus
    def create_document_corpus(self, txt_dim_df = None, txt_lang = None, 
            text_column = 'text', sample_no = None):

        if txt_dim_df is None:
            raise ValueError('Text and dimensions dataframe should be provided.')

        if txt_lang is None:
            raise ValueError('Text majority language should be provided.')

        txt_dim_df[text_column] = txt_dim_df[text_column].astype(str)

        if sample_no is not None:
            df_domain_sample_df = txt_dim_df.sample(sample_no)
        else:
            df_domain_sample_df = txt_dim_df

        # concatenate all text blocks in a list
        doc_sample = [str(doc) for doc in df_domain_sample_df[text_column]]
        #print(len(doc_sample))

        nlp = self.__build_extraction_pipe(txt_lang, with_NER = False)
        doc_corpus = textacy.Corpus(nlp, data = doc_sample)

        N = len(doc_corpus)
        NW = doc_corpus.n_tokens

        #100 000 is good, 1M is overkill
        print(" Total number of tokens in the document corpus: ", NW)
        print(" Sampled over a number of documents: ", N)

    # load a spacy pipeline: overwrite tokenization 
    # to protect hashtags(H) and mentions(@)
    def __build_extraction_pipe(self, lang, with_NER = False):
        model = None
        if lang == 'en':
            model = "en_core_web_sm"
        elif lang == 'fr':
            model = "fr_core_news_sm"
        elif lang == 'de':
            model = "de_core_news_sm"
        elif lang == 'it':
            model = "it_core_news_sm"
        elif lang == 'es':
            model = "es_core_news_sm"
        else:
            raise ValueError('Unsupported language.')

        if with_NER:
            nlp = spacy.load(model,disable = ("parser"))
        else:
            nlp = spacy.load(model,disable = ("parser", "ner"))
        nlp.add_pipe("emoji", first = True)

        # protect hashtags and mentions
        # get default pattern for tokens that don't get split
        re_token_match = _get_regex_pattern(nlp.Defaults.token_match)
        # add your patterns (here: hashtags and in-word hyphens)
        re_token_match = f"({re_token_match}|@[A-Za-z]+|#[A-Za-z]+|[A-Za-z]+-[A-Za-z]+)"
        # overwrite token_match function of the tokenizer
        nlp.tokenizer.token_match = re.compile(re_token_match).match

        # add attribute ruler with exception for hahstags
        ruler = nlp.get_pipe("attribute_ruler")
        patterns = [[{"TEXT": {"REGEX":r"[\#|\@][A-Za-z]+"}}]]

        # the attributes to assign to the matched token
        attrs = {"POS": "NOUN",'TAG':"HTG"}
        # Add rules to the attribute ruler
        ruler.add(patterns = patterns, attrs = attrs, index = 0)  # "The" in "The Who"

        return nlp

