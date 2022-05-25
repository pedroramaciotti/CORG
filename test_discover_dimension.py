# linate class

from corg import DiscoverDimension

def main():

    # fit the model and return train error
    compute_train_error = True
    model = DiscoverDimension(compute_train_error = compute_train_error)

    #dimension_file_header_names = None # no header : first column is entity (node ID)
    #dimension_file_header_names = {'entity' : 'twitter_id'} # must have at least an 'entity' column
    dimension_file_header_names = {'entity' : 'twitter_id', 'dimensions' :
            ['ca_component_0', 'ca_component_2', 'ca_component_5', 'ca_component_9']} 
                                                 # can optionally choose subset of dimensions
    X = model.load_dimensions_from_file('data/benchmark_dim_data/benchmark_data_input.csv',
            dimension_file_header_names = dimension_file_header_names)
    #print(X.shape)

    #label_file_header_names = None # no header : first column is entity, second column is label
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

    # Format 1 of decision boundary
    print('Decision boundary', model.model_decision_boundary_)

    # Format 2 of decision boundary
    print('Unit normal of decision hyperplane', model.decision_hyperplane_unit_normal)
    print('Projection of origin to hyperplane', model.origin_projection_to_hyperplane)

if __name__ == "__main__":
    main()
