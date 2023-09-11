# DFR_MIL A framework for predicting drug failure risk with optimal positive threshold determination.

## Package Dependency
pandas: 1.4.2
numpy: 1.21.5
scipy 1.7.3
scikit-learn 1.0.2

Step 1: Data Processing
python data_prepare.py

Step 2: run the code
python main.py -dataset600 True -compare True -l 0.001 -h 50 -e 100 -f 5

description
-dataset600   choose dataset
-compare      compare with other loss function
-l            learning rate
-h            hidden_num'
-e            epochs
-f            feature_num

Step 3: partial result figures
python Figures.py

Citation
To be added...
