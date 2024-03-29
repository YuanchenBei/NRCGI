# NRCGI
This is the Pytorch-version code of NRCGI ([Non-Recursive Cluster-Scale Graph Interacted Model for Click-Through Rate Prediction, CIKM2023](https://dl.acm.org/doi/10.1145/3583780.3615180)).

## Requirements
python >= 3.8

pytorch == 1.9.1

pickle == 0.7.5

scikit-learn == 0.24.2

pandas == 1.3.3

numpy == 1.21.2

tqdm == 4.62.2

## Folder Content
* In **data** folder, we provide two experimental datasets (MovieLens, Amazon-Electronics), which have been divided and clustered. Note that the Bi-Louvain clustering preprocess is adopted [the code that Liang Feng et.al provided](https://github.com/THUfl12/bipartite-louvain), thanks for their work.

  Because the dataset is too large to be directly uploaded (exceeds git's 100M file upload limitation), we store the preprocessed datasets in Google Cloud Disk, the access links are as follows:

  - **MovieLens**: <https://drive.google.com/file/d/1dCexkPr-KFR0zm1d2IojJWbTfaSjXpJJ/view?usp=sharing>

  - **Amazon-Electronics**: <https://drive.google.com/file/d/1_7Lymuov_-8OjcumFhM_BWNIEvKmhzzP/view?usp=sharing>

* In **model** folder, we provide the NRCGI source code demo and its running script file.

## Usage
The below running way is based on you have **entered the model folder and the datasets have been downloaded and placed on ./data.**
* We provide a running script file *run.sh* in **model** folder, which can run directly in bash.
```
bash run.sh
```

* You can also run a single dataset we provided directly.

For MovieLens dataset, you can use the following run command:
```
python main.py --model_name nrcgi --dataset_name ml-10m --learning_rate 0.005 --weight_decay 0.0004
```

For the Amazon-Electronics dataset, you can use the following run command:
```
python main.py --model_name nrcgi --dataset_name electronics --learning_rate 0.005 --weight_decay 0.00005
```

## Citation
```
@inproceedings{bei2023non,
  title={Non-Recursive Cluster-Scale Graph Interacted Model for Click-Through Rate Prediction},
  author={Bei, Yuanchen and Chen, Hao and Chen, Shengyuan and Huang, Xiao and Zhou, Sheng and Huang, Feiran},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={3748--3752},
  year={2023}
}
```
