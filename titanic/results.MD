### 1. Using best classifiers:
    train: 0.831, test: 0.824, best classifier cross: 0.799 (+-0.067=0.732), min (test/cross): 0.799, best classifier:
	
### 2. Taking Random Forest without weights:
    Stats: Best classifiers + Voting train: 0.852, test: 0.828, best classifier cross: 0.807 (+-0.07=0.737), min (test/cross): 0.807, best classifier:
Conclusion: Doesn't seem to make much of a difference, but slightly better

### 3. Splitting back between SibSp and Parch
    Logistic - liblinear - Stats: Default params cross: 0.795 (+-0.065=0.729), grid train: 0.795, test: 0.833, best classifier cross: 0.795 (+-0.065=0.729), min (test/cross): 0.795, best classifier:
    KNN - 14             - Stats: Default params cross: 0.786 (+-0.058=0.728), grid train: 0.786, test: 0.824, best classifier cross: 0.786 (+-0.058=0.728), min (test/cross): 0.786, best classifier:
    SVM - rbf            - Stats: Default params cross: 0.819 (+-0.063=0.756), grid train: 0.819, test: 0.855, best classifier cross: 0.819 (+-0.063=0.756), min (test/cross): 0.819, best classifier:
    SVM - poly           - Stats: Default params cross: 0.807 (+-0.062=0.745), grid train: 0.811, test: 0.833, best classifier cross: 0.811 (+-0.062=0.75), min (test/cross): 0.811, best classifier:
    NB                   - Stats: Default params cross: 0.759 (+-0.049=0.71), grid train: 0.759, test: 0.769, best classifier cross: 0.759 (+-0.049=0.71), min (test/cross): 0.759, best classifier:
    RandomForest - 9     - Stats: Default params cross: 0.832 (+-0.062=0.77), grid train: 0.829, test: 0.819, best classifier cross: 0.829 (+-0.062=0.768), min (test/cross): 0.819, best classifier:
    FINAL                                        - Stats: Best classifiers + Voting train: 0.848, test: 0.851, best classifier cross: 0.805 (+-0.074=0.731), min (test/cross): 0.805, best classifier:
Conclusion: doesn't seem to make much of a difference, leaving separately since putting together doesn't seem to help
	