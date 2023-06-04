
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_name = 'MIND'

model_name = 'NRMS'
assert model_name in [
    'NRMS', 'NAML', 'DKN', 'DFM', 'DFM_CONTENT', 'NCF', 'NGCF', 'DFM_ID', 'DFM_CONTENT_ONLY']

emb_model = 'NRMS' # default 'NRMS'
class BaseConfig():
    """
    General configurations appiled to news recommendation models
    """
    num_epochs = 2
    num_batches_show_loss = 100  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 2000
    batch_size = 128
    learning_rate = 0.0001
    num_workers = 4 # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 1
    entity_freq_threshold = 2
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 1
    dropout_probability = 0.2

    # large version MIND dataset
    num_words = 1 + 101232  #updated by data_preprocess.py
    num_categories = 1 + 18
    num_subcategories = 1 + 285
    num_entities = 1 + 21842
    num_users = 1 + 711222

    word_embedding_dim = 300
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200

class FMConfig():
    batch_size = 2048
    learning_rate = 0.001
    weight_decay = 1e-6
    epoch = 100

class NGCFConfig():
    batch_size = 1024
    epoch = 100

class MINDConfig():
    user_feat_idx = [0]
    item_feat_idx = [1, 2, 3]
    user_idx = [0]
    item_idx = [1]
    user_item_idx = [0, 1]
    field_dims = [711223, 101528, 19, 286]
    id_field_dims = [711223, 101528]
    news_emb_dim = 300
    if emb_model == 'DKN':
        news_emb_dim = 150

class NRMSConfig(BaseConfig):
    dataset_attributes = {"news": ['category', 'subcategory', 'title'], "record": []}
    # For multi-head self-attention
    num_attention_heads = 15

class NAMLConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract'],
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3


class DKNConfig(BaseConfig):
    dataset_attributes = {"news": ['category', 'subcategory', 'title', 'title_entities'], "record": []}
    # For CNN
    num_filters = 50
    window_sizes = [2, 3, 4]
    # TODO: currently context is not available
    use_context = False



