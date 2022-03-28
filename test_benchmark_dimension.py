# linate class

from corg import BenchmarkDimension

def main():

    # fit the model and return train error
    compute_train_error = False
    model = BenchmarkDimension(compute_train_error = compute_train_error)

    #dimension_file_header_names = None # no header : first column is entity (node ID)
    dimension_file_header_names = {'entity' : 'twitter_id'} # must have at least an 'entity' column
    X = model.load_dimension_from_file('data/benchmark_dim_data/benchmark_data_input.csv',
            dimension = 'ca_component_1', dimension_file_header_names = dimension_file_header_names)
    #print(X)

    #label_file_header_names = None # no header : first column is entity, second column is label
    label_file_header_names = {'entity':'twitter_id', 'label':'label'}
    Y = model.load_label_from_file('data/benchmark_dim_data/benchmark_data_parameters.csv.original',
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

    # Labels should be binary and do not need to be 0/1:
    # like "Democrat" "republican", or "left" "right"

    # logistic(beta_0,beta_1)
    print('beta0', model.beta0_)
    print('beta1', model.beta1_)

if __name__ == "__main__":
    main()
