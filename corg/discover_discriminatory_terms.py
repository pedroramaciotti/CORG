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

pattern_NP = [{"POS": "ADJ", "OP": "*"}, {"POS": {'IN': ["NOUN", 'PROPN']}, "OP": "+"}, 
        {"POS": "ADJ", "OP": "*"}, {"POS": {'IN':["CC",'ADP','NUM','DET']}, "OP": "*"},
        {"POS": "ADJ", "OP": "*"}, {"POS": {'IN':["NOUN",'PROPN']}, "OP": "*"}]

class DiscriminatoryTermsExtractor:

    def load_text_and_dimensions(self, text_and_dimensions_filename = None):
        if text_and_dimensions_filename is None:
            raise ValueError('Text and dimensions filename should be provided.')

        if not os.path.isfile(text_and_dimensions_filename):
            raise ValueError('Text and dimensions filename does not exist.')

        df = pd.read_csv(text_and_dimensions_filename, on_bad_lines = 'skip')

        return (df)

    # create a document corpus and return most frequent terms
    def extract_frequent_terms(self, txt_dim_df = None, txt_lang = None, 
            text_column = 'text', sample_no = None): # if sample_no not None sample 
                                                     # documents to create the corpus
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

        ngrmin = 1
        ngrmax = 3
        with_NER = False
        type_pattern = ['NP','VP']
        NER_types = ['PER','ORG']
        sample_dictionary, sample_index = self.__extract_terms(doc_corpus, lang = txt_lang, ngrmin = ngrmin,
                                                 ngrmax = ngrmax, type_pattern = type_pattern,
                                                 NER_extract = with_NER, NER_types = NER_types)

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

    # extract terms from corpus based on given patterns (POS NER etc), return top N based on frequency
    def __extract_terms(self, doc_corpus, lang = 'en', ngrmin = 1, ngrmax = 10, type_pattern = ['NP'], NER_extract = False,
            remove_emoji = True, NER_types = {"PER", "ORG", "GPE",'LOC'}, starting = None):

        sample_dictionary = {}
        sample_containing_doc_ids = {}

        pattern = [] 
        authorizez_ending_tags = []
        if 'NP' in type_pattern:
            pattern.append(pattern_NP)

            if lang == 'en':
                authorizez_ending_tags.extend(['NOUN','PROPN'])
            else:
                authorizez_ending_tags.extend(['NOUN','PROPN','ADJ'])
        if 'VG' in type_pattern:
            pattern.append(pattern_VG)
            authorizez_ending_tags.extend(['VERB','ADV'])

        if 'HT' in type_pattern:
            pattern.append(pattern_HT)
            authorizez_ending_tags.extend(['NOUN','PROPN'])

        if NER_extract:
            authorizez_ending_tags.extend(['NOUN','PROPN'])
            authorizez_ending_tags = set(authorizez_ending_tags)

        if len(type_pattern) == 0:
            if NER_extract:
                doc_terms = (textacy.extract.terms(doc, ents = partial(textacy.extract.entities,include_types = NER_types)) 
                        for doc in doc_corpus)
        if len(type_pattern) > 0:
            if NER_extract:
                doc_terms = (chain(textacy.extract.token_matches(doc, pattern), 
                    textacy.extract.terms(doc, ents = partial(textacy.extract.entities,include_types = NER_types)))
                    for doc in corpus)
            else:
                doc_terms = (textacy.extract.token_matches(doc, pattern) for doc in doc_corpus)


        nb_index = 0
        for doc_id, doc in enumerate(doc_terms):
            already_seen_spans = {}
            for ws in doc:

                if len(ws) > 0:

                    # first check that the pattern was not met before
                    unique_signature = str(doc_id) +  '_' + str(ws.start) + '_'  + str(ws.end)
                    if unique_signature in already_seen_spans:
                        pass
                    else:
                        already_seen_spans[unique_signature] = True

                        # check whether the starting character feature is met
                        if starting == None:
                            pass
                        else:
                            clause = False
                            for start in starting:
                                if ws.text[0][:len(start)] == start:
                                    clause = True
                            if not clause:
                                break

                        # control for the POS of the last word, and imit terms such as emoji's etc
                        if self.__common_post_mistake(ws[-1]) in authorizez_ending_tags and self.__word_to_keep(ws,
                                remove_emoji = remove_emoji, remove_http = True, remove_too_small = True):

                            ws_full = []

                            # number of hashtags and max size of words in the multiterm
                            hashtag_no, maxsize = self.__count_hashtags_and_length(ws)

                            if hashtag_no <= 1 and maxsize > 1:
                                for w in ws:
                                    wl = w.lemma_.lower()

                                    # remove special characters, such as '!', ':' etc
                                    if wl[-1] in '[!,\.:;"\'«»\-]' and w.tag_ == 'HTG':
                                        if len(wl) >= 3:
                                            wl = wl[:-1]

                                    if wl[0] in '[!,\.:;"\'«»\-]':
                                        if len(wl) >= 3:
                                            wl = wl[1:]
                                    
                                    # also remove '#' and '@'
                                    ws_full.append(wl.replace('@', '').replace('#', ''))

                                ws_full = [x for x in ws_full if not len(x) == 0]

                                if len(ws_full) >= ngrmin and len(ws_full) <= ngrmax:
                                    ws_full.sort()
                                    ws_full_ordered = ' '.join(ws_full)
                                    if len(ws_full_ordered) > 1:
                                        if not ws_full_ordered in sample_dictionary:
                                            sample_dictionary[ws_full_ordered] = {}
                                        wst = ws.text.strip()
                                        sample_dictionary[ws_full_ordered][wst] = sample_dictionary[ws_full_ordered].get(wst, 0) + 1
                                        sample_containing_doc_ids.setdefault(ws_full_ordered, []).append(doc_id)
                                        nb_index = nb_index + 1

        print()
        print(len(sample_containing_doc_ids),' candidate terms extracted.')
        print (nb_index, ' total term occurrences.')

        threshold = 1
        sample_dictionary, sample_containing_doc_ids = self.__filter_terms_by_frequency(sample_dictionary,
                sample_containing_doc_ids, threshold = threshold)

        print()
        print(len(sample_containing_doc_ids),' candidate terms remained after frequency filtering.')

        return sample_dictionary, sample_containing_doc_ids

    def __common_post_mistake(self, wrd):
        if wrd.text in ['-','.','\n','/']:
            return 'PUNKT'
        elif wrd.text in ["l'","l▒~@~Y","d'","d▒~@~Y"]:
            return 'DET'
        elif wrd.text[0] in ['@','#']:
            return 'NOUN'
        else:
            return wrd.pos_
    
    # check whether the term/word should be maintained
    def __word_to_keep(self, wrd, remove_emoji = True,
            remove_http = True, remove_too_small = True):
        #remove EMOJIs from the list by default
        if remove_emoji:
            if wrd._.has_emoji:
                return False

        #remove URLs from the list by default
        if remove_http:
            if 'http' in str(wrd):
                return False

        if remove_too_small:
            if len(wrd.text) < 2:
                return False

        return True

    def __count_hashtags_and_length(self, wrd):
        hstgs_no = 0
        wrd_size = []

        for w in wrd:
            wrd_size.append(len(w))
            if w.tag_=='HTG':
                hstgs_no = hstgs_no + 1

        return hstgs_no, max(wrd_size)

    def __filter_terms_by_frequency(self, sample_dictionary, sample_index, threshold = 2):
        sample_dictionary_filtered = {}
        sample_index_filtered = {}

        for t in list(sample_dictionary.keys())[:]:
            N = sum(sample_dictionary[t].values())

            if N >= threshold:
                sample_dictionary_filtered[t] = sample_dictionary[t]

        for t in sample_index:
            if t in sample_dictionary_filtered:
                sample_index_filtered[t] = sample_index[t]

        return sample_dictionary_filtered, sample_index_filtered
