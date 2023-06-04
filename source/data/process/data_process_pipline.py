
from data_split_val import split_val
from data_preprocess import process_data, parse_behaviors_origin
from data_dedup import merge_and_dedup

from os import path
import os

def main():

    split_val("../MIND/origin_val/behaviors.tsv", "../MIND/val/behaviors.tsv",
             "../MIND/test/behaviors.tsv", val_rate=0.5)

    # process data
    process_data()

    # deduplicate
    merge_and_dedup("../MIND/test/behaviors_no_new.tsv", "../MIND/test/behaviors_merge.tsv", "../MIND/test/behaviors_merge_dedup.tsv")

    train_dir = '../MIND/train'
    val_dir = '../MIND/val'
    test_dir = '../MIND/test'

    parse_behaviors_origin(path.join(test_dir, 'behaviors_merge_dedup.tsv'),
                    path.join(test_dir, 'behaviors_merge_dedup_parsed.tsv'),
                    path.join(train_dir, 'user2int.tsv'),
                    mode='test')

    # rename
    os.rename(path.join(val_dir, "behaviors.tsv"), path.join(val_dir, "origin_behaviors.tsv"))
    os.rename(path.join(test_dir, "behaviors.tsv"), path.join(test_dir, "origin_behaviors.tsv"))
    os.rename(path.join(val_dir, "behaviors_no_new.tsv"), path.join(val_dir, "behaviors.tsv"))
    os.rename(path.join(test_dir, "behaviors_no_new.tsv"), path.join(test_dir, "behaviors.tsv"))

if __name__ == '__main__':
    main()