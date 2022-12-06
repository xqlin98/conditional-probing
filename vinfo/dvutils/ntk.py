from copy import deepcopy

import jax
import jax.numpy as np
import numpy as onp
import scipy
from tqdm import tqdm


def get_full_kernel_matrix(input_1, input_2, kernel_fn, class_num):
    len_1 = input_1.shape[0]
    len_2 = input_2.shape[0]
    kernel_matrix = kernel_fn(input_1, input_2, 'ntk')
    kernel_matrix_full = np.tensordot(kernel_matrix, np.eye(class_num), axes=0)
    kernel_matrix_full = np.transpose(kernel_matrix_full, (0,2, 1, 3)).reshape(class_num*len_1,class_num*len_2)
    return kernel_matrix_full, kernel_matrix

def compute_score(alpha, beta, kernel_matrix_0, kernel_matrix_1, kernel_matrix_2):
    score_0 = np.matmul(np.matmul(alpha.transpose(), kernel_matrix_0), alpha)
    score_1 = np.matmul(np.matmul(beta.transpose(), kernel_matrix_1), beta)
    score_2 = np.matmul(np.matmul(alpha.transpose(), kernel_matrix_2), beta)
    score = score_0 + score_1 - 2 * score_2
    return score

def construct_dataset(train_x, train_y, index, mean, std, class_num=10):
    train_x_original = train_x[index]
    train_x_trans = train_x_original.reshape(len(index),-1)
    train_x_trans = (train_x_trans - mean)/std
    train_y_number = train_y[index]
    train_y_trans = jax.nn.one_hot(train_y_number, class_num).reshape(-1)
    return train_x_trans, train_y_trans

def linear_solver(A, A_full, b, class_num, mu=0):
    dim_A = A.shape[0]
    if dim_A > 10000:
        result = onp.linalg.solve(A_full, b)
    else:
        ridge_A = A + mu * np.eye(dim_A)
        inv_A = scipy.linalg.pinv(ridge_A)
        inv_A_full = np.tensordot(inv_A, np.eye(class_num), axes=0)
        inv_A_full = np.transpose(inv_A_full, (0,2, 1, 3)).reshape(class_num*dim_A,class_num*dim_A)
        result = np.matmul(inv_A_full,  b)
        del inv_A, inv_A_full
    return result

def linear_solver_incremental(A, pre_A_inv, b, class_num, mu=0):
    dim_A = A.shape[0]
    if dim_A == 1:
        inv_A = np.linalg.inv(A + mu * np.eye(dim_A))
    else:
        ridge_A = A + mu * np.eye(dim_A)
        block_B = ridge_A[0:(dim_A-1),(dim_A-1):(dim_A)]
        block_C = ridge_A[(dim_A-1):(dim_A), 0:(dim_A-1)]
        block_D = ridge_A[(dim_A-1):(dim_A), (dim_A-1):(dim_A)]
        
        upper_left = pre_A_inv + pre_A_inv @ block_B @ np.linalg.inv(block_D - block_C @ pre_A_inv @ block_B) @ block_C @ pre_A_inv
        upper_right = - pre_A_inv @ block_B @ np.linalg.inv(block_D - block_C @ pre_A_inv @ block_B)
        lower_left =  - np.linalg.inv(block_D - block_C @ pre_A_inv @ block_B) @ block_C @ pre_A_inv
        lower_right = np.linalg.inv(block_D - block_C @ pre_A_inv @ block_B)
        inv_A = onp.block([[upper_left, upper_right], [lower_left, lower_right]])
    inv_A_full = np.tensordot(inv_A, np.eye(class_num), axes=0)
    inv_A_full = np.transpose(inv_A_full, (0,2, 1, 3)).reshape(class_num*dim_A,class_num*dim_A)
    result = np.matmul(inv_A_full,  b)
    del inv_A_full
    return result, inv_A

def compute_score_from_datasets(dataset_full, dataset_exclude, kernel_fn, class_num=10, inverse_method="pinv"):
    train_x_full, train_y_full =  dataset_full
    train_x_exclude, train_y_exclude = dataset_exclude

    kernel_matrix_full, kernel_matrix_ori = get_full_kernel_matrix(train_x_full, train_x_full, kernel_fn, class_num)
    # inv_kernel_matrix_full = scipy.linalg.pinv(kernel_matrix_full)
    # alpha = np.matmul(inv_kernel_matrix_full, train_y_full.reshape(-1,1))
    alpha = linear_solver(kernel_matrix_ori, kernel_matrix_full, train_y_full.reshape(-1,1), class_num)
    
    new_kernel_matrix_1, new_kernel_matrix_1_ori = get_full_kernel_matrix(train_x_exclude, train_x_exclude, kernel_fn, class_num)
    # new_inv_kernel_matrix = scipy.linalg.pinv(new_kernel_matrix_1)
    # beta = np.matmul(new_inv_kernel_matrix, train_y_exclude.reshape(-1,1))
    beta = linear_solver(new_kernel_matrix_1_ori, new_kernel_matrix_1, train_y_exclude.reshape(-1,1), class_num)
    
    new_kernel_matrix_2, _ = get_full_kernel_matrix(train_x_full, train_x_exclude, kernel_fn, class_num)
    score = compute_score(alpha, beta, kernel_matrix_full, new_kernel_matrix_1, new_kernel_matrix_2)
    
    del kernel_matrix_full, new_kernel_matrix_1, new_kernel_matrix_2, _
    return score

def remove_num_samples_per_class(label_index, num_sample):
    label_index = deepcopy(label_index)
    class_num = len(label_index)
    num_per_class = num_sample // class_num
    left_num = num_sample % class_num
    num_all_class = [num_per_class]*class_num
    left_class = onp.random.choice(range(class_num), 1)[0]
    num_all_class[left_class] = num_all_class[left_class] + left_num
    new_label_index = []
    for i, class_index in enumerate(label_index):
        new_class_index = onp.random.choice(class_index, len(class_index) - num_all_class[i], replace=False).tolist()
        new_label_index.append(new_class_index)
    return new_label_index

def same_label_number_to_score(train_x, train_y, kernel_fn, dp_per_class, rand_index_list, inspect_label, validate_label, inspect_index, x_mean, x_std):
    # compute the value for multiple construction of dataset (vary the number of dp with the same label)
    scores_same_label = []
    num_dp = range(0,dp_per_class+1,10)

    for num_same_label in tqdm(num_dp):
        # try:
        sample_inspect_index = onp.random.choice(rand_index_list[inspect_label], num_same_label, replace=False).tolist()
        other_label_index = deepcopy(rand_index_list)
        other_label_index.pop(inspect_label)
        other_label_index = remove_num_samples_per_class(other_label_index, num_same_label)
        # other_label_index.pop(validate_label)
        other_label_index = onp.hstack(other_label_index).tolist()
        dataset_exclude_index = other_label_index + sample_inspect_index
        dataset_index = dataset_exclude_index + [inspect_index]
        
        dataset = construct_dataset(train_x, train_y, dataset_index, x_mean, x_std, class_num=10)
        dataset_exclude = construct_dataset(train_x, train_y, dataset_exclude_index, x_mean, x_std, class_num=10)
    
        score = compute_score_from_datasets(dataset, dataset_exclude, kernel_fn, class_num=10)
        scores_same_label.append(score[0][0])
        print(score[0][0])
        # except:
        #     scores_same_label.append(onp.nan)
    return scores_same_label, num_dp


def different_label_number_to_score(train_x, train_y, kernel_fn, dp_per_class, rand_index_list, inspect_label, validate_label, inspect_index, x_mean, x_std):
    # compute the value for multiple construction of dataset (vary the number of dp with different label)
    scores_different_label = []
    num_dp = range(0,dp_per_class+1,10)
    for num_same_label in tqdm(num_dp):
        try:
            sample_inspect_index = onp.random.choice(rand_index_list[validate_label], num_same_label, replace=False).tolist()
            other_label_index = deepcopy(rand_index_list)
            other_label_index.pop(inspect_label)
            other_label_index.pop(validate_label)
            other_label_index = onp.hstack(other_label_index).tolist()
            dataset_exclude_index = other_label_index + sample_inspect_index
            dataset_index = dataset_exclude_index + [inspect_index]
            
            dataset = construct_dataset(train_x, train_y, dataset_index, x_mean, x_std, class_num=10)
            dataset_exclude = construct_dataset(train_x, train_y, dataset_exclude_index, x_mean, x_std, class_num=10)
        
            score = compute_score_from_datasets(dataset, dataset_exclude, kernel_fn, class_num=10)
            scores_different_label.append(score[0][0])
        except:
            scores_different_label.append(onp.nan)
    return scores_different_label, num_dp
