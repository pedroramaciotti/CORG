"""accepts a csv file of the form text|dimension_1|dimension_2|dimension_n
   and returns a scsv file of the form lemma|perplexity
"""

import os

import pandas as pd
pd.set_option("display.max_columns", None)

import spacy
from spacy.tokenizer import _get_regex_pattern
from spacy.matcher import PhraseMatcher

import re

import textacy

from tqdm import tqdm

import math

from operator import itemgetter

pattern_NP = [{"POS": "ADJ", "OP": "*"}, {"POS": {'IN': ["NOUN", 'PROPN']}, "OP": "+"}, 
        {"POS": "ADJ", "OP": "*"}, {"POS": {'IN':["CC",'ADP','NUM','DET']}, "OP": "*"},
        {"POS": "ADJ", "OP": "*"}, {"POS": {'IN':["NOUN",'PROPN']}, "OP": "*"}]

class DiscriminatoryTermsExtractor:

    important_terms_df = None
    doc_terms = None
    doc_term_index = None

    def load_text_and_dimensions(self, text_and_dimensions_filename = None):
        if text_and_dimensions_filename is None:
            raise ValueError('Text and dimensions filename should be provided.')

        if not os.path.isfile(text_and_dimensions_filename):
            raise ValueError('Text and dimensions filename does not exist.')

        df = pd.read_csv(text_and_dimensions_filename, on_bad_lines = 'skip')

        return (df)

    # create a document corpus and return most frequent terms
    def extract_important_terms(self, txt_dim_df = None, txt_lang = None, 
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
        sample_corpus = textacy.Corpus(nlp, data = doc_sample)

        N = len(sample_corpus)
        NW = sample_corpus.n_tokens

        #100 000 is good, 1M is overkill
        print(" Total number of tokens in the document corpus: ", NW)
        print(" Sampled over a number of documents: ", N)

        ngrmin = 1
        ngrmax = 3
        with_NER = False
        type_pattern = ['NP','VP']
        NER_types = ['PER','ORG']
        sample_dictionary, sample_index = self.__extract_terms(sample_corpus, lang = txt_lang, ngrmin = ngrmin,
                                                 ngrmax = ngrmax, type_pattern = type_pattern,
                                                 NER_extract = with_NER, NER_types = NER_types)

        threshold = 1
        sample_dictionary, sample_index = self.__filter_terms_by_frequency(sample_dictionary,
                sample_index, threshold = threshold)

        nested, n_dict = self.__build_nested_terms(sample_dictionary)

        freq_thres = .00001
        topn = 500
        ranking = 'C-value' # other possible rankings: 'pigeon', 'tfidf'
        self.important_terms_df, sample_index = self.__select_important_terms(sample_corpus, nested, 
                sample_dictionary, sample_index, n_dict, freq_thres = freq_thres,
                topn = int(100 * topn), ranking = ranking)

        # index documents based on above terms
        sample_dictionary_main = dict(zip(self.important_terms_df['lemma'],
            self.important_terms_df['main_word']))
        sample_dictionary = dict(zip(self.important_terms_df['lemma'],
            self.important_terms_df['words']))
        for x in sample_dictionary:
            sample_dictionary[x] = eval(str(sample_dictionary[x]))

        case_insensitive = True
        all_docs = txt_dim_df[text_column]
        nlp, matcher = self.__build_indexation_pipe(txt_lang, case_insensitive = case_insensitive)
        matcher, minimal_query = self.__feed_matcher(nlp, matcher, sample_dictionary, sample_dictionary_main)
        #
        self.doc_terms, term_index, _ = self.__index_terms(all_docs, nlp, matcher)

        self.important_terms_df = self.important_terms_df.loc[self.important_terms_df['lemma'].
                isin(self.doc_terms.keys())]

        self.doc_term_index = {}   # convert doc index numbers to doc IDs
        for k in tqdm(term_index.keys()):
            self.doc_term_index[k] = []
            for d in term_index[k]:
                d_id = str(txt_dim_df.iloc[[d]]['id'].values[0])
                self.doc_term_index[k].append(d_id)

        return (self.important_terms_df)

    # given (1) an axis/dimension and (2) a subset of the dimension columns, 
    # compute the projection of each document to (1) using (2) as its actual dimensions
    def project_documents_to_dimension(self, txt_dim_df = None, projection_direction = None,
            projection_position = None, dimension_columns = None):
        if txt_dim_df is None:
            raise ValueError('Text and dimensions dataframe should be provided.')

        if projection_direction is None:
            raise ValueError('List representing the projection direction should be provided.')

        if not (type(projection_direction) is list):
            raise ValueError('Object representing the projection direction should be a list.')

        if len(projection_direction) == 0:
            raise ValueError('List representing the projection direction should not be empty.')

        # convert to list of floats
        map(float, projection_direction)

        if projection_position is None:
            raise ValueError('List representing the projection position should be provided.')

        if not (type(projection_position) is list):
            raise ValueError('Object representing the projection position should be a list.')

        if len(projection_position) == 0:
            raise ValueError('List representing the projection position should not be empty.')

        map(float, projection_position)

        if (len(projection_direction) != len(projection_position)):
            raise ValueError('Lists representing projection directions and projection positions should have same length.')

        if dimension_columns is None:
            raise ValueError('List representing the dimension columns should be provided.')

        if not (type(dimension_columns) is list):
            raise ValueError('Object representing the dimension columns should be a list.')

        if (len(projection_direction) != len(dimension_columns)):
            raise ValueError('Lists representing dimension columns and projection dimension should have same length.')

        # identify documents to project and keep only their dimension columns


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

        term_dictionary = {}
        term_containing_doc_ids = {}

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
                                        if not ws_full_ordered in term_dictionary:
                                            term_dictionary[ws_full_ordered] = {}
                                        wst = ws.text.strip()
                                        term_dictionary[ws_full_ordered][wst] = term_dictionary[ws_full_ordered].get(wst, 0) + 1
                                        term_containing_doc_ids.setdefault(ws_full_ordered, []).append(doc_id)
                                        nb_index = nb_index + 1

        print()
        print(len(term_containing_doc_ids),' candidate terms extracted.')
        print (nb_index, ' total term occurrences.')

        return term_dictionary, term_containing_doc_ids

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

    def __filter_terms_by_frequency(self, term_dictionary, term_index, threshold = 2):
        term_dictionary_filtered = {}
        term_index_filtered = {}

        for t in list(term_dictionary.keys())[:]:
            N = sum(term_dictionary[t].values())

            if N >= threshold:
                term_dictionary_filtered[t] = term_dictionary[t]

        for t in term_index:
            if t in term_dictionary_filtered:
                term_index_filtered[t] = term_index[t]

        print()
        print(len(term_index_filtered),' remaining candidate terms after frequency pre-filtering.')

        return term_dictionary_filtered, term_index_filtered

    def __build_nested_terms(self, term_dictionary):
        #print(term_dictionary)

        number_of_tokens_per_term = {}  # term, number of tokens per term
        n_dict_inv = {} # number of tokens, terms
        for cle in term_dictionary:
            n = len(cle.split())
            number_of_tokens_per_term[cle] = n
            # print(cle, n)
            n_dict_inv.setdefault(n,[]).append(cle)
        #print(n_dict_inv)

        ns = list(n_dict_inv.keys()) # sort number of tokens/term
        ns.sort()
        #print (ns)

        cless = {}  # split each term into its tokens
        for x in list(term_dictionary.keys()):
            cless[x] = set(x.split())
        #print(cless)

        print()
        Nn = len(ns)
        nested = {}
        for i in range(Nn - 1):
            n = ns[Nn - i - 1]
            #print(i, n)

            print("Finding nested strings of size:", n)
            larger = n_dict_inv[n]
            smaller = []
            for k in range(2, Nn - i + 1):
                nminus = ns[Nn - i - k]
                #print(nminus)
                smaller.extend(n_dict_inv[nminus])
            #print()
            #print("smaller", smaller)
            #print()
            #print("larger", larger)

            for x in tqdm(smaller, total = len(smaller)):
                xs = cless[x]
                #print(xs)

                for y in larger:
                    #print(y)
                    if xs <= cless[y]:
                        #print(xs, '------', cless[y])
                        nested.setdefault(x,[]).append(y)
                        if y in nested:
                            nested[x].extend(nested[y])

            #break

        #for k in nested.keys():
        #    print(k, nested[k])

        return nested, number_of_tokens_per_term

    def __select_important_terms(self, doc_corpus, nested_terms, term_dictionary, 
            term_index, n_dict, freq_thres = .0001, topn = 100, ranking = 'pigeon'):

        N = len(doc_corpus)
        NW = doc_corpus.n_tokens

        cpigeon, pigeon, freqn, cvalues, tfidf ={}, {}, {}, {}, {}
        term_frequency = {} # number of term appearances
        term_doc_no = {} # number of unique documents the term appears
        for w in term_index:
            d = len(set(term_index[w]))
            f = len(term_index[w])
            term_frequency[w] = f
            term_doc_no[w] = d

            fn = f / NW
            dth = N - N * math.pow(float((N - 1) / N), f)

            if fn > freq_thres:
                freqn[w] = fn  # term proportion in document corpus
                pigeon[w] = dth / d  # term pigeon measure
                tfidf[w] = f * math.log(N / d)
                n = len(w)

                if not w in nested_terms:
                    cvalue = (math.log2(n) + .1) * f
                else:
                    fnested = 0
                    for nested_term in nested_terms[w]:
                        fnested += len(term_index[nested_term])
                    cvalue = (math.log2(n) + 0.1) * (f - fnested / len(nested_terms[w]))
                cvalues[w] = cvalue
                cpigeon[w] = cvalue * pigeon[w]

        if ranking == 'pigeon':
            print (len(pigeon),' short-listed after the frequency proportion threshold filter ')
            key_terms_list = list(map(lambda x:x[0], sorted(pigeon.items(), key = itemgetter(1), reverse=True)))[:topn]
            print (len(key_terms_list), ' short-listed after pigeon ')
        elif ranking == 'C-value':
            print (len(cvalues),' short-listed after the frequency proportion threshold filter ')
            key_terms_list = list(map(lambda x:x[0], sorted(cvalues.items(), key = itemgetter(1), reverse = True)))[:topn]
            print (len(key_terms_list), ' short-listed after c-value ')
        elif ranking == 'tfidf':
            print (len(tfidf),' short-listed after the frequency proportion threshold filter ')
            key_terms_list = list(map(lambda x:x[0], sorted(tfidf.items(), key = itemgetter(1), reverse = True)))[:topn]
            print (len(key_terms_list), ' short-listed after tfidf ')
        else:
            raise ValueError('Term filtering criterion should be provided.')
        #print(key_terms_list)

        sample_dictionary_main = {}
        for x in key_terms_list:
            sample_dictionary_main[x] = sorted(term_dictionary[x].items(),key = itemgetter(1), reverse = True)[0][0]

        # construct result data frame
        word_list_df = pd.DataFrame(key_terms_list, columns = ['lemma'])
        word_list_df['words'] = word_list_df['lemma'].map(term_dictionary)
        word_list_df['main_word'] = word_list_df['lemma'].map(sample_dictionary_main)
        word_list_df['normalized frequency'] = word_list_df['lemma'].map(freqn)
        word_list_df['documents'] = word_list_df['lemma'].map(freqn)

        word_list_df['pigeon'] = word_list_df['lemma'].map(pigeon)

        if ranking == 'C-value' or ranking == 'C-pigeon':
            word_list_df['C-value'] = word_list_df['lemma'].map(cvalues)
            word_list_df['C-pigeon'] = word_list_df['lemma'].map(cpigeon)
        else:
            word_list_df['weighted (ngr) frequency'] = word_list_df['lemma'].map(cvalues)
            word_list_df['weighted (ngr) pigeon'] = word_list_df['lemma'].map(cpigeon)

        word_list_df['tfidf'] = word_list_df['lemma'].map(tfidf)
        word_list_df['n'] = word_list_df['lemma'].map(n_dict)

        word_list_df['occurrences'] = word_list_df['lemma'].map(term_frequency)
        word_list_df['documents'] = word_list_df['lemma'].map(term_doc_no)

        return word_list_df, sample_dictionary_main

    def __build_indexation_pipe(self, lang, case_insensitive=True):
        model = None
        if lang == 'en':
            model = "en_core_web_sm"
        elif lang == 'fr':
            model = "fr_core_news_sm"
        elif lang == 'it':
            model = "it_core_news_sm"
        elif lang == 'de':
            model = "de_core_news_sm"
        elif lang == 'es':
            model = "es_core_news_sm"

        nlp = spacy.load(model, disable = ('parser', 'ner', 'tok2vec',
            'tagger', 'morphologizer', 'lemmatizer'))
        nlp.add_pipe('sentencizer')
        nlp.add_pipe("emoji", first = True)

        # protect hashtags and mentions
        # get default pattern for tokens that don't get split
        re_token_match = _get_regex_pattern(nlp.Defaults.token_match)

        # add your patterns (here: hashtags and in-word hyphens)
        # re_token_match = f"({re_token_match}|@\w+|#\w+|\w+-\w+)"
        re_token_match = f"({re_token_match}|@[A-Za-z]+|#[A-Za-z]+|[A-Za-z]+-[A-Za-z]+)"

        # overwrite token_match function of the tokenizer
        nlp.tokenizer.token_match = re.compile(re_token_match).match

        if case_insensitive:
            matcher = PhraseMatcher(nlp.vocab, attr = 'LOWER')
        else:
            matcher = PhraseMatcher(nlp.vocab)

        return nlp, matcher

    def __feed_matcher(self, nlp, matcher, sample_dictionary, sample_dictionary_main):
        minimal_query = {}

        for cle in sample_dictionary_main:
            words = list(sample_dictionary[cle].keys())

            Nounphrases_list = self.__substring(words)

            minimal_query[cle] = Nounphrases_list

            patterns = [nlp.make_doc(Nounphrases) for Nounphrases in Nounphrases_list]

            matcher.add(cle, patterns)

        return matcher, minimal_query

    def __substring(self, string_list):
        string_list.sort(key = lambda s: len(s))
        out = []
        for s in string_list:
            if not any([self.__sub_string(o, s) for o in out]):
                out.append(s)
        return out

    def __sub_string(self, o, s):
        if o in s:
            os = set(o.split())
            ss = set(s.split())
            if os.issubset(ss):
                return True

    def __index_terms(self, docs, nlp, matcher, nested = {}):
        count = {}
        count_doc = {}
        rows = []
        lihgtrows = []
        forms_dict = {}

        for i, couple in tqdm(enumerate(docs), total = len(docs)):
            text = couple
            doc_id = i

            doc = nlp(str(text).replace("▒~@~Y","'"))

            for sent_id, sent in enumerate(doc.sents):
                matches = matcher(sent)

                found_matches = []
                for match_id, start, end in matches:
                    found_matches.append(nlp.vocab.strings[match_id])
                found_matches = set(found_matches)

                for match_id, start, end in matches:

                    span = doc[start:end]
                    match_id_string = nlp.vocab.strings[match_id]

                    if len(set(nested.get(match_id_string,[])) & found_matches) == 0:
                        count_doc.setdefault(match_id_string,[]).append(doc_id)
                        count[match_id_string] = count.get(match_id_string, 0) + 1

                        if not match_id_string in forms_dict:
                            forms_dict[match_id_string]={}
                        forms_dict[match_id_string][span.text] = forms_dict[match_id_string].get(span.text, 0) + 1

        return (count, count_doc, forms_dict)
