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
    important_terms_df = None
    if 'text_column' in params['discriminatory_terms'].keys():
        important_terms_df = dte.extract_important_terms(txt_dim_df = txt_dim_df,
                txt_lang =  params['discriminatory_terms']['text_language'],
                text_column = params['discriminatory_terms']['text_column'], sample_no = 20)
    else:
        important_terms_df = dte.extract_important_terms(txt_dim_df = txt_dim_df, 
                txt_lang =  params['discriminatory_terms']['text_language'], sample_no = 20000)
    #print(important_terms_df.columns)

    projection_axis = params['discriminatory_terms']['projection_dimension'].split(':')
    map(float, projection_axis)
    print(projection_axis)
    #dte.project_documents_to_dimension(txt_dim_df = txt_dim_df)

if __name__ == "__main__":
    main()
