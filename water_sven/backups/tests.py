###testing
import numpy as np
import pandas as pd
import time
import tensorflow as tf


def dfs3():
    df_in = pd.DataFrame({
        'col1': ['hello', 'hello', 'how', 'are', 'how', 'how'],
        'col2': [1, 2, 3, 4, 5, 6],
        'col3': [2, 2, 3, 1, 3, 3],
        'col4': ['a', 'a', 'a', 'b', 'a', 'a']
    })
    df = df_in.copy()
    print(df)
    gr = df.groupby('col1')
    for x in gr:
        print(x)
    sume = gr['col3'].transform('sum')
    print('sum', sume)
    weights = df['col3'] / df.groupby('col1')['col3'].transform('sum')
    print('w', weights)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].multiply(weights, axis=0)
    print('z', df)
    # df[numeric_cols] = df[numeric_cols].sum(axis=1)
    # print('2', df)
    df = df.groupby('col1').sum()
    print(df)

    # combined_df_adm2
    add_df = df_in.copy()
    add_df = add_df[['col1', 'col3']].copy()
    clusters = add_df['col1'].value_counts().reset_index()
    clusters.columns = ['col1', 'clusters']
    hh = add_df.groupby('col1').sum().reset_index()
    print('hh', hh)
    hh.columns = ['col1', 'households sum']
    df = pd.merge(df, hh, how='outer', on='col1')
    df = pd.merge(df, clusters, how='outer', on='col1')
    print('addhh\n', df)
    print('clusters', clusters)
    print('hh', hh)



def dfs2():
    # create a sample dataframe with string and numeric columns
    df = pd.DataFrame({
        'col1': ['hello', 'world', 'how', 'are', 'you'],
        'col2': [1, 2, 3, 4, 5],
        'col3': [0.5, 1.5, 2.5, 3.5, 4.5]
    })

    # select only the numeric columns
    num_cols = df.select_dtypes(include=['int', 'float']).columns

    # multiply the numeric columns by 2
    df = df * 2

    # combine the original and modified dataframes
    result = pd.concat([df[num_cols], df.drop(columns=num_cols)], axis=1)
    print(df)
    print(result)



def dfs():
    df = pd.DataFrame({'A': [1,2,3,4], 'B':[5,6,7,8]})
    print(df)
    print('0', df.iloc[0])
    print('0:', df.iloc[0:])
    print(':0', df.iloc[:0])
    print(':,0', df.iloc[:,0])
    print('1,0', df.iloc[1, 0])


dfs3()

def ds_creation():
    images = tf.data.Dataset.from_tensor_slices(
        tf.random.uniform([10, 3, 10, 10], minval=1, maxval=10, dtype=tf.int32))
    labels = [1,2,3,4,5,6,7,8,9.,0]
    labels = pd.Series(labels)
    print(labels)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    print(labels)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    print(images, images.element_spec)
    print('l', labels, labels.element_spec)
    dataset3 = tf.data.Dataset.zip((images, labels))
    print(dataset3)
    print(dataset3.element_spec)
    ds = dataset3
    for x in ds:
        print('One DS element in2', x)
        break


# dataset = tf.data.Dataset.from_tensor_slices((images, labels))
# dataset = dataset.batch(batch_size)
# <TensorSliceDataset element_spec=TensorSpec(shape=(10,), dtype=tf.int32, name=None)>
#
# for z in dataset1:
#   print(z.numpy())
#
# [1 6 9 2 8 7 1 9 2 4]
# [2 1 5 2 6 9 1 4 5 3]
# [8 5 6 6 7 4 5 1 8 4]
# [8 5 8 1 9 3 9 2 9 3]
#

#
# dataset2
#
# <TensorSliceDataset element_spec=(TensorSpec(shape=(), dtype=tf.float32, name=None), TensorSpec(shape=(100,), dtype=tf.int32, name=None))>
#
# dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
#



def numpy_list_comp():
    a_arr = np.array([[[1,1],
                       [1,1],
                       [1,1]],
                      [[2, 2],
                      [2, 2],
                      [2, 2]],
                     [[3,3],
                       [3,3],
                        [3,3]]])

    m_arr = np.array([1,10,100])
    a_df = pd.DataFrame(a_arr[0])
    print(a_arr)
    print(a_arr.shape)
    print(a_df)
    print(a_df.shape)

    print(m_arr)
    print(m_arr.shape)
    print('asd')

    t1 = time.time()
    out_arr = np.array([])
    for m, arr in zip(m_arr, a_arr):
        r_arr = arr * m
        out_arr = np.append(out_arr, r_arr)
    print(time.time() - t1)
    print('floop', out_arr)

    t1 = time.time()
    out_arr = np.array([xi*mi for xi, mi in zip(a_arr, m_arr)])
    print(time.time() - t1)
    print('list comp', out_arr)

# out_array = np.array([])
# for i in enumerate(replace_nan_value):
#     out_array = np.append(out_array, np.nan_to_num(array[i], nan=replace_nan_value[i]))
# array = out_array

