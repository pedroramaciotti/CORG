"""accepts a csv file of the form text|dimension_1|dimension_2|dimension_n
   and returns a scsv file of the form lemma|perplexity
"""

import os

import pandas as pd
pd.set_option("display.max_columns", None)

class DiscriminatoryTermsExtractor():

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
        print(len(doc_sample))

        #nlp = extacy.build_extraction_pipe(txt_lang, with_NER = False)
        #corpus = textacy.Corpus(nlp, data = documents_sample)

