# linate class

from corg import SingleCADimensionBenchmark

def main():

    # Note from 18th March meeting:
    # Naming "SingleCADimensionBenchmark" should "BenchmarkDimension"

    # fit the model and return train error
    compute_train_error = False
    model = SingleCADimensionBenchmark(compute_train_error = compute_train_error)

    ca_dimension_file_header_names = None # no header : first column is entity (node ID)
    ca_dimension_file_header_names = {'entity' : 'twitter_id'} # must have at least an 'entity' column
    X = model.load_CA_dimension_from_file('data/benchmark_dim_data/benchmark_data_input.csv',
            ca_dimension = 'ca_component_1', ca_dimension_file_header_names = ca_dimension_file_header_names)
    #print(X)

    label_file_header_names = None # no header : first column is entity, second column is label
    label_file_header_names = {'entity':'twitter_id', 'label':'label'}
    Y = model.load_label_from_file('data/benchmark_dim_data/benchmark_data_parameters.csv',
            label_file_header_names = label_file_header_names)
    #print(Y)

    # Note from 18th March meeting:
    # Just checking: load_label_from_file loads a file with two columns
    # One column is entity, other column is label

    # Note from 18th March meeting:
    # Labels should be binary: like "Democrat" "republican", or "left" "right"

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

    # Note from 18th March meeting:
    # Labels should be binary: like "Democrat" "republican", or "left" "right"

    # logistic(beta_0,beta_1)
    model.beta0_
    model.beta1_


if __name__ == "__main__":
    main()
