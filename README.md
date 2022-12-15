# Pessimistic Decision-Making for Recommender Systems
This repository holds the source code for [our paper "Pessimistic Reward Models for Off-Policy Learning in Recommendation" published at RecSys 2021](http://adrem.uantwerpen.be/bibrem/pubs/JeunenRecSys2021_A.pdf).
This work has been extended as "Pessimistic Decision-Making for Recommender Systems", published in the [ACM Transactions on Recommender Systems (ToRS)](https://dl.acm.org/doi/10.1145/3568029) journal.


## Reproducibility
- Install RecoGym requirements as described in the [official repository](https://github.com/criteo-research/reco-gym).
- Install utilities and common packages: TQDM, NumPy/SciPy, Pandas.
- Run `python3 src/PessimisticRidgeRegression.py`, which will run all experiments and dump raw measurements in .csv files, along with coarse visualisation in .png files.


## Paper
If you use our code in your research, please remember to cite our work:

```BibTeX
    @article{JeunenTORS_2022,
      author = {Jeunen, Olivier and Goethals, Bart},
      title = {Pessimistic Decision-Making for Recommender Systems},
      journal = {Transactions on Recommender Systems (TORS)},
      year = {2022},
      publisher = {ACM},
    }

    @inproceedings{JeunenRecSys2021_A,
      author = {Jeunen, Olivier and Goethals, Bart},
      title = {Pessimistic Reward Models for Off-Policy Learning in Recommendation},
      booktitle = {Proceedings of the 15th ACM Conference on Recommender Systems},
      series = {RecSys '21},
      year = {2021},
      publisher = {ACM},
    }
```
