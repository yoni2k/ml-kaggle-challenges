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
Conclusion: doesn't seem to make much of a difference, leaving separately since putting together doesn't seem to help. Not final.

### 4. Not removing fare outliers
	Logistic - liblinear - Stats: Default params cross: 0.801 (+-0.074=0.726), grid train: 0.801, test: 0.812, best classifier cross: 0.801 (+-0.074=0.726), min (test/cross): 0.801, best classifier:
	KNN - 14             - Stats: Default params cross: 0.798 (+-0.058=0.74), grid train: 0.798, test: 0.803, best classifier cross: 0.798 (+-0.058=0.74), min (test/cross): 0.798, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.822 (+-0.054=0.768), grid train: 0.826, test: 0.834, best classifier cross: 0.826 (+-0.058=0.769), min (test/cross): 0.826, best classifier:
	SVM - poly           - Stats: Default params cross: 0.808 (+-0.056=0.752), grid train: 0.808, test: 0.843, best classifier cross: 0.808 (+-0.056=0.753), min (test/cross): 0.808, best classifier:
	NB                   - Stats: Default params cross: 0.79 (+-0.065=0.726), grid train: 0.79, test: 0.785, best classifier cross: 0.79 (+-0.065=0.726), min (test/cross): 0.785, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.813 (+-0.046=0.767), grid train: 0.816, test: 0.816, best classifier cross: 0.816 (+-0.04=0.775), min (test/cross): 0.816, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.835, test: 0.807, best classifier cross: 0.816 (+-0.059=0.757), min (test/cross): 0.807, best classifier:
Conclusion: seems slightly better both cross, min, although test is much lower.  Not final.

### 5. Not removing Embarked feature
	KNN - 14             - Stats: Default params cross: 0.793 (+-0.073=0.721), grid train: 0.793, test: 0.821, best classifier cross: 0.793 (+-0.073=0.721), min (test/cross): 0.793, best classifier:
	Logistic - liblinear - Stats: Default params cross: 0.799 (+-0.077=0.723), grid train: 0.799, test: 0.812, best classifier cross: 0.799 (+-0.077=0.723), min (test/cross): 0.799, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.813 (+-0.058=0.755), grid train: 0.819, test: 0.825, best classifier cross: 0.819 (+-0.064=0.754), min (test/cross): 0.819, best classifier:
	SVM - poly           - Stats: Default params cross: 0.816 (+-0.061=0.755), grid train: 0.819, test: 0.821, best classifier cross: 0.819 (+-0.057=0.762), min (test/cross): 0.819, best classifier:
	NB                   - Stats: Default params cross: 0.783 (+-0.082=0.7), grid train: 0.783, test: 0.789, best classifier cross: 0.783 (+-0.082=0.7), min (test/cross): 0.783, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.813 (+-0.043=0.77), grid train: 0.816, test: 0.816, best classifier cross: 0.816 (+-0.037=0.778), min (test/cross): 0.816, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.843, test: 0.807, best classifier cross: 0.81 (+-0.068=0.742), min (test/cross): 0.807, best classifier:
Conclusion: not final, doesn't seem to make much of a difference, slightly slightly better to remove, so removing. Not final

### 6. Removing Fare altogether
	Logistic - liblinear - Stats: Default params cross: 0.796 (+-0.079=0.718), grid train: 0.796, test: 0.83, best classifier cross: 0.796 (+-0.079=0.718), min (test/cross): 0.796, best classifier:
	KNN - 14             - Stats: Default params cross: 0.795 (+-0.06=0.734), grid train: 0.795, test: 0.803, best classifier cross: 0.795 (+-0.06=0.734), min (test/cross): 0.795, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.825 (+-0.055=0.769), grid train: 0.825, test: 0.834, best classifier cross: 0.825 (+-0.059=0.765), min (test/cross): 0.825, best classifier:
	SVM - poly           - Stats: Default params cross: 0.81 (+-0.061=0.748), grid train: 0.811, test: 0.843, best classifier cross: 0.811 (+-0.063=0.748), min (test/cross): 0.811, best classifier:
	NB                   - Stats: Default params cross: 0.762 (+-0.069=0.693), grid train: 0.762, test: 0.78, best classifier cross: 0.762 (+-0.069=0.693), min (test/cross): 0.762, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.816 (+-0.052=0.764), grid train: 0.82, test: 0.821, best classifier cross: 0.82 (+-0.06=0.76), min (test/cross): 0.82, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.829, test: 0.812, best classifier cross: 0.819 (+-0.06=0.758), min (test/cross): 0.812, best classifier:
Conclusion: much better to remove Fare altogether than have it as is.  Remove for now, than see if can return in a better way.

### 7. Removing Age altogether
	Logistic - liblinear - Stats: Default params cross: 0.796 (+-0.07=0.726), grid train: 0.796, test: 0.807, best classifier cross: 0.796 (+-0.07=0.726), min (test/cross): 0.796, best classifier:
	KNN - 14             - Stats: Default params cross: 0.798 (+-0.044=0.753), grid train: 0.798, test: 0.771, best classifier cross: 0.798 (+-0.044=0.753), min (test/cross): 0.771, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.798 (+-0.058=0.74), grid train: 0.801, test: 0.812, best classifier cross: 0.801 (+-0.072=0.729), min (test/cross): 0.801, best classifier:
	SVM - poly           - Stats: Default params cross: 0.786 (+-0.058=0.728), grid train: 0.796, test: 0.798, best classifier cross: 0.796 (+-0.055=0.741), min (test/cross): 0.796, best classifier:
	NB                   - Stats: Default params cross: 0.76 (+-0.072=0.689), grid train: 0.76, test: 0.78, best classifier cross: 0.76 (+-0.072=0.689), min (test/cross): 0.76, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.784 (+-0.056=0.729), grid train: 0.786, test: 0.785, best classifier cross: 0.786 (+-0.056=0.729), min (test/cross): 0.785, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.811, test: 0.803, best classifier cross: 0.802 (+-0.058=0.744), min (test/cross): 0.802, best classifier:
Conclusion: seems worse, returning

### 8. Removing SibSp feature
	Logistic - liblinear - Stats: Default params cross: 0.798 (+-0.081=0.717), grid train: 0.798, test: 0.794, best classifier cross: 0.798 (+-0.081=0.717), min (test/cross): 0.794, best classifier:
	KNN - 14             - Stats: Default params cross: 0.801 (+-0.068=0.733), grid train: 0.801, test: 0.803, best classifier cross: 0.801 (+-0.068=0.733), min (test/cross): 0.801, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.81 (+-0.075=0.735), grid train: 0.81, test: 0.812, best classifier cross: 0.81 (+-0.072=0.738), min (test/cross): 0.81, best classifier:
	SVM - poly           - Stats: Default params cross: 0.804 (+-0.073=0.731), grid train: 0.804, test: 0.816, best classifier cross: 0.804 (+-0.073=0.731), min (test/cross): 0.804, best classifier:
	NB                   - Stats: Default params cross: 0.78 (+-0.066=0.714), grid train: 0.78, test: 0.758, best classifier cross: 0.78 (+-0.066=0.714), min (test/cross): 0.758, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.79 (+-0.062=0.729), grid train: 0.792, test: 0.816, best classifier cross: 0.792 (+-0.064=0.728), min (test/cross): 0.792, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.823, test: 0.807, best classifier cross: 0.813 (+-0.073=0.74), min (test/cross): 0.807, best classifier:
Conclusion: slightly worse without, put back

### 9. Removing Parch feature
	Logistic - liblinear - Stats: Default params cross: 0.795 (+-0.076=0.718), grid train: 0.795, test: 0.812, best classifier cross: 0.795 (+-0.076=0.718), min (test/cross): 0.795, best classifier:
	KNN - 14             - Stats: Default params cross: 0.795 (+-0.061=0.734), grid train: 0.795, test: 0.821, best classifier cross: 0.795 (+-0.061=0.734), min (test/cross): 0.795, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.82 (+-0.058=0.763), grid train: 0.825, test: 0.825, best classifier cross: 0.825 (+-0.054=0.77), min (test/cross): 0.825, best classifier:
	SVM - poly           - Stats: Default params cross: 0.814 (+-0.059=0.755), grid train: 0.822, test: 0.83, best classifier cross: 0.822 (+-0.064=0.757), min (test/cross): 0.822, best classifier:
	NB                   - Stats: Default params cross: 0.757 (+-0.072=0.685), grid train: 0.757, test: 0.78, best classifier cross: 0.757 (+-0.072=0.685), min (test/cross): 0.757, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.829 (+-0.051=0.778), grid train: 0.829, test: 0.821, best classifier cross: 0.829 (+-0.06=0.769), min (test/cross): 0.821, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.844, test: 0.816, best classifier cross: 0.822 (+-0.06=0.762), min (test/cross): 0.816, best classifier:
Conclusion: seems better to remove it, although only slightly. Not final

### 10. Removing Sex to prove that it's needed
	Logistic - liblinear - Stats: Default params cross: 0.671 (+-0.033=0.638), grid train: 0.671, test: 0.722, best classifier cross: 0.671 (+-0.033=0.638), min (test/cross): 0.671, best classifier:
	KNN - 14             - Stats: Default params cross: 0.688 (+-0.028=0.66), grid train: 0.688, test: 0.776, best classifier cross: 0.688 (+-0.028=0.66), min (test/cross): 0.688, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.706 (+-0.036=0.67), grid train: 0.706, test: 0.753, best classifier cross: 0.706 (+-0.036=0.67), min (test/cross): 0.706, best classifier:
	SVM - poly           - Stats: Default params cross: 0.693 (+-0.028=0.665), grid train: 0.698, test: 0.735, best classifier cross: 0.698 (+-0.029=0.668), min (test/cross): 0.698, best classifier:
	NB                   - Stats: Default params cross: 0.654 (+-0.039=0.615), grid train: 0.654, test: 0.686, best classifier cross: 0.654 (+-0.039=0.615), min (test/cross): 0.654, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.676 (+-0.032=0.645), grid train: 0.688, test: 0.74, best classifier cross: 0.688 (+-0.025=0.664), min (test/cross): 0.688, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.723, test: 0.758, best classifier cross: 0.696 (+-0.037=0.659), min (test/cross): 0.696, best classifier:
Conclusion: Indeed needed

### 11. Removing Pclass to prove that it's needed
	Logistic - liblinear - Stats: Default params cross: 0.793 (+-0.076=0.717), grid train: 0.793, test: 0.785, best classifier cross: 0.793 (+-0.076=0.717), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.801 (+-0.059=0.742), grid train: 0.801, test: 0.807, best classifier cross: 0.801 (+-0.059=0.742), min (test/cross): 0.801, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.82 (+-0.064=0.756), grid train: 0.822, test: 0.812, best classifier cross: 0.822 (+-0.066=0.756), min (test/cross): 0.812, best classifier:
	SVM - poly           - Stats: Default params cross: 0.789 (+-0.073=0.716), grid train: 0.792, test: 0.789, best classifier cross: 0.792 (+-0.074=0.718), min (test/cross): 0.789, best classifier:
	NB                   - Stats: Default params cross: 0.793 (+-0.068=0.726), grid train: 0.793, test: 0.807, best classifier cross: 0.793 (+-0.068=0.726), min (test/cross): 0.793, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.792 (+-0.062=0.729), grid train: 0.793, test: 0.785, best classifier cross: 0.793 (+-0.055=0.738), min (test/cross): 0.785, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.819, test: 0.821, best classifier cross: 0.813 (+-0.072=0.741), min (test/cross): 0.813, best classifier:
Conclusion: Surprising, but it's not clear cut that it's needed.  Leaving for now. Not final.

### 12. Staying only with 'Sex' and 'Age' is not enough
	Logistic - liblinear - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.777 (+-0.071=0.705), grid train: 0.777, test: 0.758, best classifier cross: 0.777 (+-0.071=0.705), min (test/cross): 0.758, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.796 (+-0.077=0.72), grid train: 0.796, test: 0.789, best classifier cross: 0.796 (+-0.077=0.72), min (test/cross): 0.789, best classifier:
	SVM - poly           - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	NB                   - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.783 (+-0.066=0.716), grid train: 0.786, test: 0.767, best classifier cross: 0.786 (+-0.067=0.719), min (test/cross): 0.767, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.792, test: 0.789, best classifier cross: 0.792 (+-0.078=0.714), min (test/cross): 0.789, best classifier:
	
### 13. Sex only - benchmark to check what to add first
	Logistic - liblinear - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	SVM - poly           - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	NB                   - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	
### 13a. Sex + Fare
	Logistic - liblinear - Stats: Default params cross: 0.786 (+-0.073=0.713), grid train: 0.786, test: 0.771, best classifier cross: 0.786 (+-0.073=0.713), min (test/cross): 0.771, best classifier:
	KNN - 14             - Stats: Default params cross: 0.759 (+-0.073=0.686), grid train: 0.759, test: 0.776, best classifier cross: 0.759 (+-0.073=0.686), min (test/cross): 0.759, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.79 (+-0.071=0.719), grid train: 0.79, test: 0.785, best classifier cross: 0.79 (+-0.071=0.719), min (test/cross): 0.785, best classifier:
	SVM - poly           - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.79, test: 0.785, best classifier cross: 0.79 (+-0.071=0.719), min (test/cross): 0.785, best classifier:
	NB                   - Stats: Default params cross: 0.783 (+-0.073=0.71), grid train: 0.783, test: 0.776, best classifier cross: 0.783 (+-0.073=0.71), min (test/cross): 0.776, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.772 (+-0.059=0.713), grid train: 0.777, test: 0.821, best classifier cross: 0.777 (+-0.065=0.712), min (test/cross): 0.777, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.79, test: 0.785, best classifier cross: 0.787 (+-0.073=0.715), min (test/cross): 0.785, best classifier:

### 13b. Sex + Parch
	Logistic - liblinear - Stats: Default params cross: 0.789 (+-0.072=0.717), grid train: 0.789, test: 0.785, best classifier cross: 0.789 (+-0.072=0.717), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.78 (+-0.078=0.702), grid train: 0.78, test: 0.785, best classifier cross: 0.78 (+-0.078=0.702), min (test/cross): 0.78, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.795 (+-0.073=0.721), grid train: 0.795, test: 0.785, best classifier cross: 0.795 (+-0.073=0.721), min (test/cross): 0.785, best classifier:
	SVM - poly           - Stats: Default params cross: 0.792 (+-0.07=0.722), grid train: 0.792, test: 0.785, best classifier cross: 0.792 (+-0.07=0.722), min (test/cross): 0.785, best classifier:
	NB                   - Stats: Default params cross: 0.789 (+-0.075=0.714), grid train: 0.789, test: 0.785, best classifier cross: 0.789 (+-0.075=0.714), min (test/cross): 0.785, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.792 (+-0.07=0.722), grid train: 0.792, test: 0.785, best classifier cross: 0.792 (+-0.07=0.722), min (test/cross): 0.785, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.795, test: 0.785, best classifier cross: 0.792 (+-0.07=0.722), min (test/cross): 0.785, best classifier:


### 13c. Sex + Embarked
	Logistic - liblinear - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	SVM - poly           - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	NB                   - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:

### 13d. Sex + Age
	Logistic - liblinear - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.777 (+-0.071=0.705), grid train: 0.777, test: 0.758, best classifier cross: 0.777 (+-0.071=0.705), min (test/cross): 0.758, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.796 (+-0.077=0.72), grid train: 0.796, test: 0.789, best classifier cross: 0.796 (+-0.077=0.72), min (test/cross): 0.789, best classifier:
	SVM - poly           - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	NB                   - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.79 (+-0.066=0.724), grid train: 0.789, test: 0.762, best classifier cross: 0.789 (+-0.071=0.718), min (test/cross): 0.762, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.792, test: 0.789, best classifier cross: 0.792 (+-0.078=0.714), min (test/cross): 0.789, best classifier:


### 13e. Sex + SibSp
	Logistic - liblinear - Stats: Default params cross: 0.793 (+-0.076=0.717), grid train: 0.793, test: 0.785, best classifier cross: 0.793 (+-0.076=0.717), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.776 (+-0.073=0.702), grid train: 0.776, test: 0.798, best classifier cross: 0.776 (+-0.073=0.702), min (test/cross): 0.776, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.789 (+-0.073=0.716), grid train: 0.793, test: 0.785, best classifier cross: 0.793 (+-0.076=0.717), min (test/cross): 0.785, best classifier:
	SVM - poly           - Stats: Default params cross: 0.789 (+-0.073=0.716), grid train: 0.792, test: 0.785, best classifier cross: 0.792 (+-0.074=0.718), min (test/cross): 0.785, best classifier:
	NB                   - Stats: Default params cross: 0.793 (+-0.068=0.726), grid train: 0.793, test: 0.807, best classifier cross: 0.793 (+-0.068=0.726), min (test/cross): 0.793, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.79 (+-0.074=0.717), grid train: 0.787, test: 0.794, best classifier cross: 0.787 (+-0.075=0.712), min (test/cross): 0.787, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.792, test: 0.798, best classifier cross: 0.79 (+-0.075=0.716), min (test/cross): 0.79, best classifier:

### 13f. Sex + Pclass
	Logistic - liblinear - Stats: Default params cross: 0.787 (+-0.075=0.713), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.781 (+-0.071=0.71), grid train: 0.781, test: 0.785, best classifier cross: 0.781 (+-0.071=0.71), min (test/cross): 0.781, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.768 (+-0.066=0.702), grid train: 0.787, test: 0.785, best classifier cross: 0.787 (+-0.075=0.713), min (test/cross): 0.785, best classifier:
	SVM - poly           - Stats: Default params cross: 0.769 (+-0.068=0.701), grid train: 0.789, test: 0.78, best classifier cross: 0.789 (+-0.048=0.741), min (test/cross): 0.78, best classifier:
	NB                   - Stats: Default params cross: 0.769 (+-0.079=0.69), grid train: 0.769, test: 0.785, best classifier cross: 0.769 (+-0.079=0.69), min (test/cross): 0.769, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.768 (+-0.066=0.702), grid train: 0.769, test: 0.78, best classifier cross: 0.769 (+-0.068=0.701), min (test/cross): 0.769, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.787, test: 0.785, best classifier cross: 0.78 (+-0.076=0.703), min (test/cross): 0.78, best classifier:
Conclusion: 	
Voting train:
- benchmark: 0.787
- Best: Parch (0.795)
- Better: SibSp, Age, Parch, Fare
- Worse/same: Pclass, Embarked
Test: 
- benchmark: 0.785
- Best: SibSp (0.798)
- Better: SibSp (0.798), Age (0.789)
- Worse/same: Pclass (0.785), Embarked (0.785), Parch (0.785), Fare (0.785)
Cross:
- benchmark: 0.787
- Best: Age (0.792), Parch(0.792)
- Better: SibSp (0.79), Age (0.792), Parch(0.792), 
- Worse/same: Pclass (0.78), Embarked (0.787), Fare (0.787)
Min cross:
- benchmark: 0.713
- Best: Parch(0.722)
- Better: SibSp (0.716), Age (0.714), Parch(0.722), Fare (0.715)
- Worse/same: Pclass (0.703), Embarked (0.713)
Best test: SibSp
Best cross: Age, then SibSp
Best min: Pclass
Adding SibSp

### 14-. (Sex + SibSp) - benchmark
	Logistic - liblinear - Stats: Default params cross: 0.793 (+-0.076=0.717), grid train: 0.793, test: 0.785, best classifier cross: 0.793 (+-0.076=0.717), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.776 (+-0.073=0.702), grid train: 0.776, test: 0.798, best classifier cross: 0.776 (+-0.073=0.702), min (test/cross): 0.776, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.789 (+-0.073=0.716), grid train: 0.793, test: 0.785, best classifier cross: 0.793 (+-0.076=0.717), min (test/cross): 0.785, best classifier:
	SVM - poly           - Stats: Default params cross: 0.789 (+-0.073=0.716), grid train: 0.792, test: 0.785, best classifier cross: 0.792 (+-0.074=0.718), min (test/cross): 0.785, best classifier:
	NB                   - Stats: Default params cross: 0.793 (+-0.068=0.726), grid train: 0.793, test: 0.807, best classifier cross: 0.793 (+-0.068=0.726), min (test/cross): 0.793, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.79 (+-0.074=0.717), grid train: 0.787, test: 0.794, best classifier cross: 0.787 (+-0.075=0.712), min (test/cross): 0.787, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.792, test: 0.798, best classifier cross: 0.79 (+-0.075=0.716), min (test/cross): 0.79, best classifier:

### 14a. (Sex + SibSp) + Embarked
	Logistic - liblinear - Stats: Default params cross: 0.793 (+-0.076=0.717), grid train: 0.793, test: 0.785, best classifier cross: 0.793 (+-0.076=0.717), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.777 (+-0.07=0.707), grid train: 0.777, test: 0.807, best classifier cross: 0.777 (+-0.07=0.707), min (test/cross): 0.777, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.787 (+-0.075=0.712), grid train: 0.793, test: 0.785, best classifier cross: 0.793 (+-0.076=0.717), min (test/cross): 0.785, best classifier:
	SVM - poly           - Stats: Default params cross: 0.789 (+-0.073=0.716), grid train: 0.789, test: 0.798, best classifier cross: 0.789 (+-0.073=0.716), min (test/cross): 0.789, best classifier:
	NB                   - Stats: Default params cross: 0.793 (+-0.068=0.726), grid train: 0.793, test: 0.807, best classifier cross: 0.793 (+-0.068=0.726), min (test/cross): 0.793, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.778 (+-0.075=0.703), grid train: 0.784, test: 0.794, best classifier cross: 0.784 (+-0.079=0.705), min (test/cross): 0.784, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.793, test: 0.807, best classifier cross: 0.787 (+-0.075=0.712), min (test/cross): 0.787, best classifier:

### 14b. (Sex + SibSp) + Fare
	Logistic - liblinear - Stats: Default params cross: 0.786 (+-0.072=0.714), grid train: 0.786, test: 0.785, best classifier cross: 0.786 (+-0.072=0.714), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.786 (+-0.069=0.717), grid train: 0.786, test: 0.776, best classifier cross: 0.786 (+-0.069=0.717), min (test/cross): 0.776, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.793 (+-0.074=0.72), grid train: 0.796, test: 0.785, best classifier cross: 0.796 (+-0.073=0.724), min (test/cross): 0.785, best classifier:
	SVM - poly           - Stats: Default params cross: 0.789 (+-0.073=0.716), grid train: 0.792, test: 0.785, best classifier cross: 0.792 (+-0.072=0.72), min (test/cross): 0.785, best classifier:
	NB                   - Stats: Default params cross: 0.795 (+-0.072=0.723), grid train: 0.795, test: 0.798, best classifier cross: 0.795 (+-0.072=0.723), min (test/cross): 0.795, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.78 (+-0.063=0.717), grid train: 0.789, test: 0.798, best classifier cross: 0.789 (+-0.062=0.727), min (test/cross): 0.789, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.798, test: 0.803, best classifier cross: 0.793 (+-0.07=0.724), min (test/cross): 0.793, best classifier:

### 14c. (Sex + SibSp) + Parch
	Logistic - liblinear - Stats: Default params cross: 0.793 (+-0.076=0.717), grid train: 0.793, test: 0.785, best classifier cross: 0.793 (+-0.076=0.717), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.787 (+-0.07=0.717), grid train: 0.787, test: 0.794, best classifier cross: 0.787 (+-0.07=0.717), min (test/cross): 0.787, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.793 (+-0.07=0.723), grid train: 0.801, test: 0.785, best classifier cross: 0.801 (+-0.075=0.726), min (test/cross): 0.785, best classifier:
	SVM - poly           - Stats: Default params cross: 0.796 (+-0.07=0.726), grid train: 0.796, test: 0.785, best classifier cross: 0.796 (+-0.07=0.726), min (test/cross): 0.785, best classifier:
	NB                   - Stats: Default params cross: 0.795 (+-0.068=0.727), grid train: 0.795, test: 0.807, best classifier cross: 0.795 (+-0.068=0.727), min (test/cross): 0.795, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.795 (+-0.065=0.73), grid train: 0.789, test: 0.785, best classifier cross: 0.789 (+-0.06=0.729), min (test/cross): 0.785, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.804, test: 0.807, best classifier cross: 0.793 (+-0.07=0.723), min (test/cross): 0.793, best classifier:

### 14d. (Sex + SibSp) + Pclass
	Logistic - liblinear - Stats: Default params cross: 0.793 (+-0.074=0.719), grid train: 0.793, test: 0.807, best classifier cross: 0.793 (+-0.074=0.719), min (test/cross): 0.793, best classifier:
	KNN - 14             - Stats: Default params cross: 0.799 (+-0.052=0.747), grid train: 0.799, test: 0.803, best classifier cross: 0.799 (+-0.052=0.747), min (test/cross): 0.799, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.801 (+-0.059=0.742), grid train: 0.801, test: 0.798, best classifier cross: 0.801 (+-0.059=0.742), min (test/cross): 0.798, best classifier:
	SVM - poly           - Stats: Default params cross: 0.796 (+-0.065=0.732), grid train: 0.798, test: 0.803, best classifier cross: 0.798 (+-0.062=0.736), min (test/cross): 0.798, best classifier:
	NB                   - Stats: Default params cross: 0.757 (+-0.073=0.685), grid train: 0.757, test: 0.78, best classifier cross: 0.757 (+-0.073=0.685), min (test/cross): 0.757, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.796 (+-0.053=0.743), grid train: 0.796, test: 0.789, best classifier cross: 0.796 (+-0.053=0.743), min (test/cross): 0.789, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.798, test: 0.803, best classifier cross: 0.795 (+-0.064=0.731), min (test/cross): 0.795, best classifier:

### 14e. (Sex + SibSp) + Age
	Logistic - liblinear - Stats: Default params cross: 0.793 (+-0.076=0.717), grid train: 0.793, test: 0.785, best classifier cross: 0.793 (+-0.076=0.717), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.801 (+-0.059=0.742), grid train: 0.801, test: 0.807, best classifier cross: 0.801 (+-0.059=0.742), min (test/cross): 0.801, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.82 (+-0.064=0.756), grid train: 0.822, test: 0.812, best classifier cross: 0.822 (+-0.066=0.756), min (test/cross): 0.812, best classifier:
	SVM - poly           - Stats: Default params cross: 0.789 (+-0.073=0.716), grid train: 0.792, test: 0.789, best classifier cross: 0.792 (+-0.074=0.718), min (test/cross): 0.789, best classifier:
	NB                   - Stats: Default params cross: 0.793 (+-0.068=0.726), grid train: 0.793, test: 0.807, best classifier cross: 0.793 (+-0.068=0.726), min (test/cross): 0.793, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.787 (+-0.062=0.725), grid train: 0.784, test: 0.798, best classifier cross: 0.784 (+-0.065=0.719), min (test/cross): 0.784, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.819, test: 0.821, best classifier cross: 0.813 (+-0.072=0.741), min (test/cross): 0.813, best classifier:
Deciding what to add to (Sex + SibSp)
Conclusion: 	
Voting train:
- benchmark: 0.792
- Best: Age (0.819)
- Better: Age (0.819), Pclass (0.798), Parch (0.804), Fare (0.798)
- Worse/same: Embarked (0.793)
Test: 
- benchmark: 0.798
- Best: Age (0.821)
- Better: Age (0.821), Pclass (0.803), Parch (0.807), Fare (0.803), Embarked (0.807)
- Worse/same: 
Cross:
- benchmark: 0.79
- Best: Age (0.813)
- Better: Age (0.813), Pclass (0.795), Parch (0.793), Fare (0.793), 
- Worse/same: Embarked (0.787)
Min cross:
- benchmark: 0.716
- Best: Age (0.741)
- Better: Age (0.741), Pclass (0.731), Parch (0.723), Fare (0.724), Embarked (0.712)
- Worse/same: 
Best test: Age
Best cross: Age
Best min: Age
Conclusion - not try Embarked - 2nd round always the worst. Add Age





Conclusion: 	
Voting train:
- benchmark: 
- Best: 
- Better: 
- Worse/same: 
Test: 
- benchmark: 
- Best: 
- Better: 
- Worse/same: 
Cross:
- benchmark: 
- Best: 
- Better: 
- Worse/same: 
Min cross:
- benchmark: 
- Best: 
- Better: 
- Worse/same: 
Best test: 
Best cross: 
Best min: 