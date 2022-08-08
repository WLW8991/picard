def fake_realtion_fulfilling(data_base_dir, dataset_name, t5_processed, mode):
    import numpy as np
    relation_matrix_l = []
    for item in t5_processed:
        relation_matrix = np.ones((len(item), len(item)))
        relation_matrix_l.append(relation_matrix)
    return relation_matrix_l

def preprocess_by_dataset(data_base_dir, dataset_name, t5_processed, mode):
    relations=fake_realtion_fulfilling(data_base_dir, dataset_name, t5_processed, mode)
    return relations