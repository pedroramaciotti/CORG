# linate class

from corg import CADimensionDiscovery

def main():


    # fit the model and return train error
    compute_train_error = False
    model = CADimensionDiscovery(compute_train_error = compute_train_error)

    #ca_dimension_file_header_names = None # no header : first column is entity (node ID)
    #ca_dimension_file_header_names = {'entity' : 'twitter_id'} # must have at least an 'entity' column
    ca_dimension_file_header_names = {'entity' : 'twitter_id', 'ca_dimensions' :
            ['ca_component_0', 'ca_component_2', 'ca_component_5', 'ca_component_10']} 
                                                 # can optionally choose subset of dimensions
    X = model.load_CA_dimensions_from_file('data/benchmark_dim_data/benchmark_data_input.csv',
            ca_dimension_file_header_names = ca_dimension_file_header_names)
    print(X)

    '''
    label_file_header_names = None # no header : first column is entity, second column is label
    label_file_header_names = {'entity':'twitter_id', 'label':'label'}
    Y = model.load_label_from_file('data/benchmark_dim_data/benchmark_data_parameters.csv',
            label_file_header_names = label_file_header_names)
    #print(Y)

    model.fit(X, Y)

    if compute_train_error:
        print(model.accuracy_train_)
        print(model.precision_train_)
        print(model.recall_train_)
        print(model.f1_score_train_)
    else: 
        print(model.accuracy_mean_)
        print(model.accuracy_std_)
        print(model.precision_mean_)
        print(model.precision_std_)
        print(model.recall_mean_)
        print(model.recall_std_)
        print(model.f1_score_mean_)
        print(model.f1_score_std_)
    '''

if __name__ == "__main__":
    main()
