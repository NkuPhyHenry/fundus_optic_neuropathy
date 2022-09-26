import numpy as np
import os
from tqdm import tqdm
import shutil

'''
生成训练集与测试集
'''


def generate_train_test(dirs, train_save, test_save, test_rt=0.05):
    if os.path.exists(train_save):
        shutil.rmtree(train_save)
    if os.path.exists(test_save):
        shutil.rmtree(test_save)
    # test_fn = 0
    # train_fn = 0
    for d in dirs:
        ids = [f for f in os.listdir(d)
               if os.path.isfile(os.path.join(d, f))]
        order = np.arange(len(ids))
        np.random.shuffle(order)  # 随机打乱数据

        n_test = int(test_rt * len(ids))  # 测试集数据
        print(f'''Splitting data into
              {len(ids) - n_test} train & 
              {n_test} test''')

        test_order = order[:n_test].tolist()  # 将数组转化成列表
        train_order = order[n_test:].tolist()

        if not os.path.exists(test_save):
            os.makedirs(test_save)
        if not os.path.exists(train_save):
            os.makedirs(train_save)

        for i in tqdm(test_order):
            shutil.copy(os.path.join(d, ids[i]), os.path.join(test_save, ids[i]))
            '''
            拷贝被打乱数据里面的多少位（一共需要的数量）
            '''
            # test_fn += 1

        for i in tqdm(train_order):
            shutil.copy(os.path.join(d, ids[i]), os.path.join(train_save, ids[i]))
            # train_fn += 1
    return True


if __name__ == '__main__':
    folders = r'D:\研一上\cnn-eye\fundus_optic_neuropathy\git_hub\original'

    classes = [1, 2, 3]
    for c in classes:
        print(f'Generating for folder {folders} class {c}')
        ori_dirs = [f'{folders}/{c}']
        generate_train_test(ori_dirs, f'D:/研一上/cnn-eye/fundus_optic_neuropathy/git_hub/data/train/{c}',
                            f'D:/研一上/cnn-eye/fundus_optic_neuropathy/git_hub/data/test/{c}', test_rt=0.2)