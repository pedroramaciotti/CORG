# DiscriminatoryTermsExtractor class

from corg import DiscriminatoryTermsExtractor

import configparser

import sys

def main():
    params = configparser.ConfigParser()  # read parameters from file
    params.read(sys.argv[1])

    dte = DiscriminatoryTermsExtractor()
    txt_dim_df = dte.load_text_and_dimensions(params['discriminatory_terms']['text_and_dimensions_file'])

    # 'text_language': en, fr, de, it, es
    if 'text_column' in params['discriminatory_terms'].keys():
        dte.extract_important_terms(txt_dim_df = txt_dim_df, txt_lang =  params['discriminatory_terms']['text_language'],
                text_column = params['discriminatory_terms']['text_column'], sample_no = 20)
    else:
        dte.extract_important_terms(txt_dim_df = txt_dim_df, 
                txt_lang =  params['discriminatory_terms']['text_language'], sample_no = 20000)

if __name__ == "__main__":
    main()
