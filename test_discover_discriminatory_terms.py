# DiscriminatoryTermsExtractor class

from corg import DiscriminatoryTermsExtractor

import configparser

import sys

def main():
    params = configparser.ConfigParser()  # read parameters from file
    params.read(sys.argv[1])

    dte = DiscriminatoryTermsExtractor()
    txt_dim_df = dte.load_text_and_dimensions(params['discriminatory_terms']['text_and_dimensions_file'])
    txt_dim_df = txt_dim_df.sample(1000)

    # 'text_language': en, fr, de, it, es
    important_terms_df = None
    if 'text_column' in params['discriminatory_terms'].keys():
        important_terms_df = dte.extract_important_terms(txt_dim_df = txt_dim_df,
                txt_lang =  params['discriminatory_terms']['text_language'],
                text_column = params['discriminatory_terms']['text_column'], sample_no = 20)
    else:
        important_terms_df = dte.extract_important_terms(txt_dim_df = txt_dim_df, 
                txt_lang =  params['discriminatory_terms']['text_language'], sample_no = 20000)

    projection_direction = params['discriminatory_terms']['projection_direction_vector'].split(':')
    map(float, projection_direction)
    #print(projection_direction)
    #
    projection_position = params['discriminatory_terms']['projection_position_vector'].split(':')
    map(float, projection_position)
    #print(projection_position)

    dimension_columns = params['discriminatory_terms']['dimension_columns'].split(':')
    #print(dimension_columns)
    dte.project_documents_to_dimension(projection_direction, projection_position, dimension_columns)

if __name__ == "__main__":
    main()
