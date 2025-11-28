import json
import numpy as np
import os

def parse_json_file(file_path):
    with open(file_path, 'rb') as file:
        datas = json.load(file)
        train_datas = datas.get('train', [])
        test_datas = datas.get('test', [])
        # train = []
        # test = []
        # for data in train_datas:
        #     train.append(parse_data(data))
        # for data in test_datas:
        #     test.append(parse_data(data))
        train = [parse_data(data) for data in train_datas]
        test = [parse_data(data) for data in test_datas]
            
    return train, test

def parse_data(data):
    input_raw = data.get('input', [])
    output_raw = data.get('output', [])
    input_array = np.array(input_raw, dtype=np.int64)
    output_array = np.array(output_raw, dtype=np.int64)
    return input_array, output_array

def test_dataset_grid():
    grid_size_limits = (3, 3)
    file_sets = set()
    for file_path in os.listdir('./data/arc2/training'):
        if file_path.endswith('.json'):
            full_path = os.path.join('./data/arc2/training', file_path)
            train, test = parse_json_file(full_path)
            # if all arrays smaller than or equal to grid_size_limits
            # add file to set
            for input_array, output_array in train + test:
                hi, wi = input_array.shape
                ho, wo = output_array.shape
                if hi > grid_size_limits[0] or wi > grid_size_limits[1] or ho > grid_size_limits[0] or wo > grid_size_limits[1]:
                    break
            else:
                file_sets.add(file_path)
                    
    print(f"Train files with grid size within {grid_size_limits}:\n {file_sets}")
    file_sets = set()
    for file_path in os.listdir('./data/arc2/evaluation'):
        if file_path.endswith('.json'):
            full_path = os.path.join('./data/arc2/evaluation', file_path)
            train, test = parse_json_file(full_path)
            for input_array, output_array in train + test:
                hi, wi = input_array.shape
                ho, wo = output_array.shape
                if hi > grid_size_limits[0] or wi > grid_size_limits[1] or ho > grid_size_limits[0] or wo > grid_size_limits[1]:
                    break
            else:
                file_sets.add(file_path)
    print(f"Evaluation files with grid size within {grid_size_limits}:\n {file_sets}")

if __name__ == "__main__":
    test_dataset_grid()


