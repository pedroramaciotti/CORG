# linate class

from corg import SingleCADimensionBenchmark

def main():

    # fit the model and return train error
    model = SingleCADimensionBenchmark(compute_train_error = True)

    ca_dimension_file_header_names = None # no header : first column is entity (node ID)
    ca_dimension_file_header_names = {'entity' : 'twitter_id'} # must have at least an 'entity' column
    X = model.load_CA_feature_from_file('data/benchmark_dim_data/benchmark_data_input.csv',
            ca_dimension = 'ca_component_1', ca_dimension_file_header_names = ca_dimension_file_header_names)
    #print(X)

    label_file_header_names = None # no header : first column is entity, second column is label
    label_file_header_names = {'entity':'twitter_id', 'label':'label'}
    Y = model.load_label_from_file('data/benchmark_dim_data/benchmark_data_parameters.csv',
            label_file_header_names = label_file_header_names)
    print(Y)

if __name__ == "__main__":
    main()
