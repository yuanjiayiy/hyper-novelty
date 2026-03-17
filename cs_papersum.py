# %%
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# conf_names = ['AAAI', 'ACL', 'AISTATS', 'ASPLOS', 'CCS', 'CHI', 'CIKM', 'COLT', 'CRYPTO', 'CVPR', 'EMNLP', 'FG', 'HPCA', 'HRI', 'ICCV', 'ICLR', 'ICME', 'ICSE', 'INFOCOM', 'ISBI', 'ISMAR', 'MICCAI', 'MICRO', 'NDSS', 'NeurIPS', 'PLDI', 'TCC', 'USENIX', 'VRIC', 'WADS', 'WWW']
# conf_data=[]
# for i in conf_names:
#     j=pd.read_csv(f'Keywords/{i}-keywords.csv')
#     conf_data.append(j)
# df = pd.concat(conf_data, ignore_index=True)
# def filter_keywords(keywords):
#     keywords_list = ast.literal_eval(keywords)
#     filtered_list = [item for item in keywords_list if item[1] > 0.5]
#     return str(filtered_list)
# df['capped_keywords'] = df['combined_keywords'].apply(filter_keywords)
# df.to_csv('All_capped_keywords.csv', index=False)

# %%
df=pd.read_csv('../Data/All_capped_keywords.csv')  

# %%
# Group conferences by field
fields = {
    'Computing in Biomedical Fields': [
        'International Conference on Medical Image Computing and Computer-Assisted Intervention',
        'IEEE International Symposium on Biomedical Imaging'
    ],
    'Natural Language Processing': [
        'Conference on Empirical Methods in Natural Language Processing',
        'Annual Meeting of the Association for Computational Linguistics'
    ],
    'Computer Vision': [
        'Computer Vision and Pattern Recognition',
        'IEEE International Conference on Computer Vision'
    ],
    'Artificial Intelligence': [
        'AAAI Conference on Artificial Intelligence',
        'International Conference on Learning Representations'
    ],
    'Computational Theory': [
        'Neural Information Processing Systems',
        'International Conference on Artificial Intelligence and Statistics',
        'Annual Conference Computational Learning Theory'
    ],
    'Computer Hardware & Architecture': [
        'Micro',
        'International Symposium on High-Performance Computer Architecture'
    ],
    'Computer Networks & Communications': [
        'The Web Conference',
        'Conference on Computer and Communications Security',
        'IEEE Conference on Computer Communications'
    ],
    'Computer Security & Cryptography': [
        'Network and Distributed System Security Symposium',
        'USENIX Annual Technical Conference',
        'Theory of Cryptography Conference',
        'Annual International Cryptology Conference'
    ],
    'Database & Information Systems': [
        'Workshop on Algorithms and Data Structures',
        'International Conference on Information and Knowledge Management'
    ],
    'Graphics and Computer-Aided Design': [
        'IEEE International Conference on Automatic Face & Gesture Recognition',
        'IEEE International Conference on Multimedia and Expo'
    ],
    'Human Computer Interaction': [
        'International Conference on Human Factors in Computing Systems',
        'IEEE/ACM International Conference on Human-Robot Interaction'
    ],
    'Mixed and Augmented Reality': [
        'Virtual Reality International Conference',
        'International Symposium on Mixed and Augmented Reality'
    ],
    'Software Engineering': [
        'International Conference on Software Engineering',
        'International Conference on Architectural Support for Programming Languages and Operating Systems',
        'ACM-SIGPLAN Symposium on Programming Language Design and Implementation'
    ]
}
def map_venue_to_field(venue):
    for field, conferences in fields.items():
        if venue in conferences:
            return field

# Apply the function to create the new column
df['field'] = df['venue'].apply(map_venue_to_field)

# %%
df.head()

# %%
from datasets import Dataset, DatasetDict, load_dataset
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index) 

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    "train": train_ds,
    "test": test_ds,
})

dataset.push_to_hub("yuancarrieyjy/CS-PaperSum", commit_message="Initial commit, from https://arxiv.org/abs/2502.20582")