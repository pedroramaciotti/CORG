# linate class

from corg import CADimensionDiscovery

def main():

    # Note from 18th March meeting:
    # Naming "CADimensionDiscovery" should "DiscoverDimension"

    # fit the model and return train error
    compute_train_error = True
    model = CADimensionDiscovery(compute_train_error = compute_train_error)

    #ca_dimension_file_header_names = None # no header : first column is entity (node ID)
    #ca_dimension_file_header_names = {'entity' : 'twitter_id'} # must have at least an 'entity' column
    ca_dimension_file_header_names = {'entity' : 'twitter_id', 'ca_dimensions' :
            ['ca_component_0', 'ca_component_2', 'ca_component_5', 'ca_component_9']} 
                                                 # can optionally choose subset of dimensions
    X = model.load_CA_dimensions_from_file('data/benchmark_dim_data/benchmark_data_input.csv',
            ca_dimension_file_header_names = ca_dimension_file_header_names)
    #print(X)

    #label_file_header_names = None # no header : first column is entity, second column is label
    label_file_header_names = {'entity':'twitter_id', 'label':'label'}
    Y = model.load_label_from_file('data/benchmark_dim_data/benchmark_data_parameters.csv',
            label_file_header_names = label_file_header_names)
    #print(Y)

    model.fit(X, Y)
    print(model.model_decision_boundary_)

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

    # Note from 18th March meeting:
    # Different return formats for the boundary decision
    #
    # Format 1: the n+1 coefficients of the hyperplane (n number of dimensions of the dataset): b0+b1*x1+b2*x2+...
    #
    # Format 2: return 2 vectors:
    #            - the unit normal of the hyperplane
    #            - and the projection of the origin on the hyperplane

if __name__ == "__main__":
    main()
