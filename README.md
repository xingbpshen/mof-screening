<h1 align="center">
Scalable and Robust Metal-organic Frameworks Screening for Carbon Capture and Storage
</h1>

This repository contains the code and data for the technical report "Scalable and Robust Metal-organic Frameworks Screening for Carbon Capture and Storage" by X. Shen and A. Saxena. This report is a part of the final course project for the _COMP 685 Machine Learning Applied to Climate Change_ at McGill University.

## Acknowledgements

The authors would like to thank Prof. David Rolnick for his guidance and support throughout the project. Also, we would like to thank the authors of the paper "Deep-Learning-Based End-to-End Predictions of CO2 Capture in Metal–Organic Frameworks" (C. Lu _et al._) and "High-performing deep learning regression models for predicting low-pressure CO2 adsorption properties of metal–organic frameworks" (J. Burner _et al._) for providing the data used in this project. The folder hierarchies and a very small portion of the code for arguments and configurations parsing is adapted from the [DDIM code base](https://github.com/ermongroup/ddim), we have properly cited the original authors in the code. We sincerely thank the authors for making their code available.

## Data

The preprocessed data used in this project can be downloaded from the following link: (comming soon)

## Requirements

We recommend using PyTorch 2.2.0 and CUDA 12.1 to avoid potential floating points computing issues (please also see this [GitHub issue](https://github.com/traveller59/spconv/issues/725)). The requirements are listed in `requirements.txt`. You can install them using the following command:
```
pip install -r requirements.txt
```

## Usage

### Modifying the configurations

The configurations for the experiments are stored in the `configs` folder. You can modify the configurations to run different experiments. In a minimum case, you need to modify the `data.data_dir` in the configuration files to point to the correct dataset path.

### Running the baseline methods

1. First, train the model using the following command:
```
python main.py --config {METHOD}.yml --exp {PROJECT_PATH} --doc {TAG} --train
```
where `METHOD` is the method's name (e.g., `baseline_a` or `baseline_b`), `PROJECT_PATH` is the path to store the experiment results (e.g., `./log`). The `--train` option is used to train the model. All training information will be stored in the `PROJECT_PATH/TAG/`.

2. Then, test the model using the following command:
```
python main.py --config {METHOD}.yml --exp {PROJECT_PATH} --doc {TAG} --test
```
where `METHOD`, `PROJECT_PATH`, and `TAG` are the same as above. The `--test` option is used to test the model. The test results will be stored in the `PROJECT_PATH/TAG/`.

### Running the proposed method

Coming soon

## References

- Lu, C., Wan, X., Ma, X., Guan, X., & Zhu, A. (2022). Deep-Learning-Based End-to-End Predictions of CO2 Capture in Metal–Organic Frameworks. Journal of Chemical Information and Modeling, 62(14), 3281-3290.
- Burner, J., Schwiedrzik, L., Krykunov, M., Luo, J., Boyd, P. G., & Woo, T. K. (2020). High-performing deep learning regression models for predicting low-pressure CO2 adsorption properties of metal–organic frameworks. The Journal of Physical Chemistry C, 124(51), 27996-28005.


