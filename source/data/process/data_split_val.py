import pandas as pd

def split_val(data_path, val_save_path, test_save_path, val_rate=0.5):
    behaviors = pd.read_table(data_path, header=None,
                names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])

    num = len(behaviors)
    val_behaviors = behaviors[ :int(val_rate*num)]
    test_behaviors = behaviors[int(val_rate*num): ] #re index
    test_behaviors.impression_id = range(1,len(test_behaviors)+1)

    val_behaviors.to_csv(val_save_path, sep="\t", header=False, index=False)
    test_behaviors.to_csv(test_save_path, sep="\t", header=False, index=False)


