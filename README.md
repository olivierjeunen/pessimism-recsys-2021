# Pessimistic Reward Models for Off-Policy Learning in Recommendation
Source code for [our paper "Pessimistic Reward Models for Off-Policy Learning in Recommendation" published at RecSys 2021](https://adrem.uantwerpen.be/bibrem/pubs/JeunenRecSys2021_A.pdf).


## Reproducibility
- Install RecoGym requirements as described in the [official repository](https://github.com/criteo-research/reco-gym).
- Install utilities and common packages: TQDM, NumPy/SciPy, Pandas.
- Run `python3 src/PessimisticRidgeRegression.py`, which will run all experiments and dump raw measurements in .csv files, along with coarse visualisation in .png files.


## Paper
If you use our code in your research, please remember to cite our paper:

```BibTeX
    @inproceedings{JeunenRecSys2021_A,
      author = {Jeunen, Olivier and Goethals, Bart},
      title = {Pessimistic Reward Models for Off-Policy Learning in Recommendation},
      booktitle = {Proceedings of the 15th ACM Conference on Recommender Systems},
      series = {RecSys '21},
      year = {2021},
      publisher = {ACM},
    }
```
