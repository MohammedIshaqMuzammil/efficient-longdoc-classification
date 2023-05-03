## Source codes for ``Efficient Classification of Long Documents Using Transformers''

Please refer to our paper for more details and cite our paper if you find this repo useful:

```
@inproceedings{park-etal-2022-efficient,
    title = "Efficient Classification of Long Documents Using Transformers",
    author = "Park, Hyunji  and
      Vyas, Yogarshi  and
      Shah, Kashif",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.79",
    doi = "10.18653/v1/2022.acl-short.79",
    pages = "702--709",
}
```

## Instructions

### 1. Install required libraries

```
conda env create -f environment.yml
conda activate elc
```

### 2. Prepare the datasets

#### CMU Book Summary Dataset

* Available at <http://www.cs.cmu.edu/~dbamman/booksummaries.html>

```
wget -P data/ http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz
tar -xf data/booksummaries.tar.gz -C data
```

* Running `train.py` with the `--data books` flag reads and prepares the data from `data/booksummaries/booksummaries.txt`
* Running `train.py` with the `--data books --pairs` flag creates Paired Book Summary by combining pairs of summaries and their labels

#### Yelp Dataset

* Available at <https://raw.githubusercontent.com/knowitall/yelp-dataset-challenge/master/data/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json>

```
wget -P data/ https://github.com/knowitall/yelp-dataset-challenge/raw/master/data/yelp_phoenix_academic_dataset/yelp_academic_dataset_review.json.tar.gz
tar -xf data/yelp_academic_dataset_review.tar.gz -C data
```

* Running `train.py` with the `--data yelp` flag reads and prepares the data from `data/yelp_academic_dataset_review.json`


### 3. Run the models

```
e.g. python train.py --model_name bertplusrandom --data books --pairs --batch_size 8 --epochs 20 --lr 3e-05
```

cf. Note that we use the source code for the CogLTX model: <https://github.com/Sleepychord/CogLTX>
