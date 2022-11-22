# DiscriminatoryTermsExtractor class

from corg import DiscriminatoryTermsExtractor

import configparser

import sys

doc_sample_no = 50000
#doc_sample_no = 200

def main():
    params = configparser.ConfigParser()  # read parameters from file
    params.read(sys.argv[1])

    histogram_bin_number = int(params['discriminatory_terms']['histogram_bin_number'])

    dte = DiscriminatoryTermsExtractor()
    txt_dim_df = dte.load_text_and_dimensions(params['discriminatory_terms']['text_and_dimensions_file'])
    #txt_dim_df = txt_dim_df.sample(1000)

    # 'text_language': en, fr, de, it, es
    important_terms_df = None
    if 'text_column' in params['discriminatory_terms'].keys():
        important_terms_df = dte.extract_important_terms(txt_dim_df = txt_dim_df,
                txt_lang =  params['discriminatory_terms']['text_language'], 
                topn = int(params['discriminatory_terms']['topN']),
                text_column = params['discriminatory_terms']['text_column'],
                sample_no = doc_sample_no, frequeny_threshold = histogram_bin_number)
    else:
        important_terms_df = dte.extract_important_terms(txt_dim_df = txt_dim_df, 
                topn = int(params['discriminatory_terms']['topN']),
                txt_lang =  params['discriminatory_terms']['text_language'],
                sample_no = doc_sample_no, frequeny_threshold = histogram_bin_number)
    #print(important_terms_df)

    # save terms and their containing documents in an edge form
    dte.save_important_term_index(params['discriminatory_terms']['term_index_file'])

    projection_direction = params['discriminatory_terms']['projection_direction_vector'].split(':')
    map(float, projection_direction)
    #print(projection_direction)
    #
    projection_position = params['discriminatory_terms']['projection_position_vector'].split(':')
    map(float, projection_position)
    #print(projection_position)

    dimension_columns = params['discriminatory_terms']['dimension_columns'].split(':')
    #print(dimension_columns)
    doc_proj_df = dte.project_documents_to_dimension(projection_direction,
            projection_position, dimension_columns)
    #print(doc_proj_df)

    term_metrics_df = dte.compute_term_perplexity_and_skewness(histogram_bins = histogram_bin_number)
    #print(term_metrics_df)
    term_metrics_df.to_csv(params['discriminatory_terms']['discriminatory_term_file'], index = False)

if __name__ == "__main__":
    main()
