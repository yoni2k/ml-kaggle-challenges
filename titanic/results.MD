### There were many different tries before, from this point working systematically
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
Best test: Age (.821)
Best cross: Age (.822)
Best min: Age (.756)
Conclusion - not try Embarked - 2nd round always the worst. Add Age

### 15-. (Sex + SibSp + Age) - benchmark
	Logistic - liblinear - Stats: Default params cross: 0.793 (+-0.076=0.717), grid train: 0.793, test: 0.785, best classifier cross: 0.793 (+-0.076=0.717), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.801 (+-0.059=0.742), grid train: 0.801, test: 0.807, best classifier cross: 0.801 (+-0.059=0.742), min (test/cross): 0.801, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.82 (+-0.064=0.756), grid train: 0.822, test: 0.812, best classifier cross: 0.822 (+-0.066=0.756), min (test/cross): 0.812, best classifier:
	SVM - poly           - Stats: Default params cross: 0.789 (+-0.073=0.716), grid train: 0.792, test: 0.789, best classifier cross: 0.792 (+-0.074=0.718), min (test/cross): 0.789, best classifier:
	NB                   - Stats: Default params cross: 0.793 (+-0.068=0.726), grid train: 0.793, test: 0.807, best classifier cross: 0.793 (+-0.068=0.726), min (test/cross): 0.793, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.787 (+-0.062=0.725), grid train: 0.784, test: 0.798, best classifier cross: 0.784 (+-0.065=0.719), min (test/cross): 0.784, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.819, test: 0.821, best classifier cross: 0.813 (+-0.072=0.741), min (test/cross): 0.813, best classifier:

### 15a. (Sex + SibSp + Age) + Fare
	Logistic - liblinear - Stats: Default params cross: 0.786 (+-0.074=0.712), grid train: 0.786, test: 0.785, best classifier cross: 0.786 (+-0.074=0.712), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.805 (+-0.063=0.742), grid train: 0.805, test: 0.794, best classifier cross: 0.805 (+-0.063=0.742), min (test/cross): 0.794, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.822 (+-0.06=0.762), grid train: 0.823, test: 0.825, best classifier cross: 0.823 (+-0.059=0.764), min (test/cross): 0.823, best classifier:
	SVM - poly           - Stats: Default params cross: 0.79 (+-0.073=0.717), grid train: 0.79, test: 0.798, best classifier cross: 0.79 (+-0.075=0.716), min (test/cross): 0.79, best classifier:
	NB                   - Stats: Default params cross: 0.793 (+-0.07=0.724), grid train: 0.793, test: 0.798, best classifier cross: 0.793 (+-0.07=0.724), min (test/cross): 0.793, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.82 (+-0.056=0.764), grid train: 0.813, test: 0.794, best classifier cross: 0.813 (+-0.048=0.765), min (test/cross): 0.794, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.822, test: 0.807, best classifier cross: 0.808 (+-0.071=0.738), min (test/cross): 0.807, best classifier:

### 15b. (Sex + SibSp + Age) + Parch
	Logistic - liblinear - Stats: Default params cross: 0.793 (+-0.076=0.717), grid train: 0.793, test: 0.785, best classifier cross: 0.793 (+-0.076=0.717), min (test/cross): 0.785, best classifier:
	KNN - 14             - Stats: Default params cross: 0.808 (+-0.066=0.743), grid train: 0.808, test: 0.812, best classifier cross: 0.808 (+-0.066=0.743), min (test/cross): 0.808, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.823 (+-0.066=0.757), grid train: 0.826, test: 0.821, best classifier cross: 0.826 (+-0.068=0.758), min (test/cross): 0.821, best classifier:
	SVM - poly           - Stats: Default params cross: 0.802 (+-0.075=0.727), grid train: 0.802, test: 0.807, best classifier cross: 0.802 (+-0.075=0.727), min (test/cross): 0.802, best classifier:
	NB                   - Stats: Default params cross: 0.795 (+-0.068=0.727), grid train: 0.795, test: 0.807, best classifier cross: 0.795 (+-0.068=0.727), min (test/cross): 0.795, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.804 (+-0.048=0.756), grid train: 0.801, test: 0.807, best classifier cross: 0.801 (+-0.049=0.752), min (test/cross): 0.801, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.828, test: 0.825, best classifier cross: 0.819 (+-0.069=0.75), min (test/cross): 0.819, best classifier:

### 15c. (Sex + SibSp + Age) + Pclass
	Logistic - liblinear - Stats: Default params cross: 0.795 (+-0.076=0.718), grid train: 0.795, test: 0.812, best classifier cross: 0.795 (+-0.076=0.718), min (test/cross): 0.795, best classifier:
	KNN - 14             - Stats: Default params cross: 0.795 (+-0.061=0.734), grid train: 0.795, test: 0.821, best classifier cross: 0.795 (+-0.061=0.734), min (test/cross): 0.795, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.82 (+-0.058=0.763), grid train: 0.825, test: 0.825, best classifier cross: 0.825 (+-0.054=0.77), min (test/cross): 0.825, best classifier:
	SVM - poly           - Stats: Default params cross: 0.814 (+-0.059=0.755), grid train: 0.822, test: 0.83, best classifier cross: 0.822 (+-0.064=0.757), min (test/cross): 0.822, best classifier:
	NB                   - Stats: Default params cross: 0.757 (+-0.072=0.685), grid train: 0.757, test: 0.78, best classifier cross: 0.757 (+-0.072=0.685), min (test/cross): 0.757, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.82 (+-0.059=0.762), grid train: 0.826, test: 0.825, best classifier cross: 0.826 (+-0.053=0.773), min (test/cross): 0.825, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.843, test: 0.821, best classifier cross: 0.823 (+-0.062=0.761), min (test/cross): 0.821, best classifier:
Summary: What to add to (Sex + SibSp + Age)	
Voting train:
- benchmark: 0.819
- Best: Pclass (0.843)
- Better: Pclass (0.843), Parch (0.828), Fare (.822)
- Worse/same: 
Test: 
- benchmark: 0.821
- Best: Parch (0.825)
- Better: Parch (0.825),  
- Worse/same: Pclass (0.821), Fare (0.807)
Cross:
- benchmark: 0.813
- Best: 
- Better: Pclass (0.823), Parch (0.819), 
- Worse/same: Fare (0.808)
Min cross:
- benchmark: 0.741
- Best: Pclass (0.761)
- Better: Pclass (0.761), Parch (0.75)
- Worse/same: Fare (0.738)
Best test: Pclass (.83)
Best cross: All similar (.825)
Best min: Pclass (.77)
Conclusion: Adding Pclass seems to help, pretty clear that Fare gets in the way, remove it

### 16. (Sex + SibSp + Age + Pclass) - add Parch ?
Benchmark:
	Logistic - liblinear - Stats: Default params cross: 0.795 (+-0.076=0.718), grid train: 0.795, test: 0.812, best classifier cross: 0.795 (+-0.076=0.718), min (test/cross): 0.795, best classifier:
	KNN - 14             - Stats: Default params cross: 0.795 (+-0.061=0.734), grid train: 0.795, test: 0.821, best classifier cross: 0.795 (+-0.061=0.734), min (test/cross): 0.795, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.82 (+-0.058=0.763), grid train: 0.825, test: 0.825, best classifier cross: 0.825 (+-0.054=0.77), min (test/cross): 0.825, best classifier:
	SVM - poly           - Stats: Default params cross: 0.814 (+-0.059=0.755), grid train: 0.822, test: 0.83, best classifier cross: 0.822 (+-0.064=0.757), min (test/cross): 0.822, best classifier:
	NB                   - Stats: Default params cross: 0.757 (+-0.072=0.685), grid train: 0.757, test: 0.78, best classifier cross: 0.757 (+-0.072=0.685), min (test/cross): 0.757, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.82 (+-0.059=0.762), grid train: 0.826, test: 0.825, best classifier cross: 0.826 (+-0.053=0.773), min (test/cross): 0.825, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.843, test: 0.821, best classifier cross: 0.823 (+-0.062=0.761), min (test/cross): 0.821, best classifier:
With Parch:
	Logistic - liblinear - Stats: Default params cross: 0.796 (+-0.079=0.718), grid train: 0.796, test: 0.83, best classifier cross: 0.796 (+-0.079=0.718), min (test/cross): 0.796, best classifier:
	KNN - 14             - Stats: Default params cross: 0.795 (+-0.06=0.734), grid train: 0.795, test: 0.803, best classifier cross: 0.795 (+-0.06=0.734), min (test/cross): 0.795, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.825 (+-0.055=0.769), grid train: 0.825, test: 0.834, best classifier cross: 0.825 (+-0.059=0.765), min (test/cross): 0.825, best classifier:
	SVM - poly           - Stats: Default params cross: 0.81 (+-0.061=0.748), grid train: 0.811, test: 0.843, best classifier cross: 0.811 (+-0.063=0.748), min (test/cross): 0.811, best classifier:
	NB                   - Stats: Default params cross: 0.762 (+-0.069=0.693), grid train: 0.762, test: 0.78, best classifier cross: 0.762 (+-0.069=0.693), min (test/cross): 0.762, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.817 (+-0.049=0.768), grid train: 0.816, test: 0.834, best classifier cross: 0.816 (+-0.052=0.764), min (test/cross): 0.816, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.829, test: 0.812, best classifier cross: 0.82 (+-0.06=0.76), min (test/cross): 0.812, best classifier:
Train:
- FINAL: worse
- Rest: same
Test:
- FINAL: worse
- Rest: not conclusive
Cross:
- FINAL: worse
- Rest: not conclusive
Min:
- FINAL: slightly worse
- Rest: slightly worse
Conclusion: not adding

### 17. (Sex + SibSp + Age + Pclass) - using all data
	Logistic - liblinear - Stats: Default params cross: 0.795 (+-0.076=0.718), grid train: 0.795, test: 0.812, best classifier cross: 0.795 (+-0.076=0.718), min (test/cross): 0.795, best classifier:
	KNN - 14             - Stats: Default params cross: 0.795 (+-0.061=0.734), grid train: 0.795, test: 0.821, best classifier cross: 0.795 (+-0.061=0.734), min (test/cross): 0.795, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.82 (+-0.058=0.763), grid train: 0.825, test: 0.825, best classifier cross: 0.825 (+-0.054=0.77), min (test/cross): 0.825, best classifier:
	SVM - poly           - Stats: Default params cross: 0.814 (+-0.059=0.755), grid train: 0.822, test: 0.83, best classifier cross: 0.822 (+-0.064=0.757), min (test/cross): 0.822, best classifier:
	NB                   - Stats: Default params cross: 0.757 (+-0.072=0.685), grid train: 0.757, test: 0.78, best classifier cross: 0.757 (+-0.072=0.685), min (test/cross): 0.757, best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.823 (+-0.063=0.76), grid train: 0.816, test: 0.816, best classifier cross: 0.816 (+-0.054=0.762), min (test/cross): 0.816, best classifier:
	FINAL                              - Stats: ALL DATA: Best classifiers + Voting train: 0.841, best classifier cross: 0.822 (+-0.051=0.77), best classifier:
Conclusion: giving same results locally, but with less STD, but on the site it gives 1 additional %

### 18. Working on whole data for all steps
	Logistic - liblinear - Stats: Default params cross: 0.8 (+-0.019=0.781), grid train: 0.81, best classifier cross: 0.8 (+-0.019=0.781), best classifier:
	KNN - 14             - Stats: Default params cross: 0.803 (+-0.037=0.766), grid train: 0.832, best classifier cross: 0.803 (+-0.037=0.766), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.824 (+-0.035=0.788), grid train: 0.834, best classifier cross: 0.826 (+-0.038=0.788), best classifier:
	SVM - poly           - Stats: Default params cross: 0.822 (+-0.032=0.79), grid train: 0.823, best classifier cross: 0.822 (+-0.032=0.79), best classifier:
	NB                   - Stats: Default params cross: 0.761 (+-0.03=0.731), grid train: 0.762, best classifier cross: 0.761 (+-0.03=0.731), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.815 (+-0.04=0.775), grid train: 0.896, best classifier cross: 0.818 (+-0.04=0.778), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.838, best classifier cross: 0.812 (+-0.029=0.782), best classifier:
Conclusion: 

### 19. Try readding Parch
	Logistic - liblinear - Stats: Default params cross: 0.8 (+-0.019=0.781), grid train: 0.81, best classifier cross: 0.8 (+-0.019=0.781), best classifier:
	KNN - 14             - Stats: Default params cross: 0.801 (+-0.037=0.764), grid train: 0.824, best classifier cross: 0.801 (+-0.037=0.764), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.824 (+-0.034=0.789), grid train: 0.836, best classifier cross: 0.827 (+-0.037=0.79), best classifier:
	SVM - poly           - Stats: Default params cross: 0.814 (+-0.034=0.78), grid train: 0.825, best classifier cross: 0.816 (+-0.036=0.78), best classifier:
	NB                   - Stats: Default params cross: 0.759 (+-0.032=0.727), grid train: 0.763, best classifier cross: 0.759 (+-0.032=0.727), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.827 (+-0.047=0.78), grid train: 0.903, best classifier cross: 0.826 (+-0.051=0.775), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.842, best classifier cross: 0.825 (+-0.032=0.793), best classifier:
Conclusion: seems to help on all fronts

### 20. Try readding Fare
	Logistic - liblinear - Stats: Default params cross: 0.803 (+-0.024=0.779), grid train: 0.806, best classifier cross: 0.803 (+-0.024=0.779), best classifier:
	KNN - 14             - Stats: Default params cross: 0.811 (+-0.035=0.777), grid train: 0.831, best classifier cross: 0.811 (+-0.035=0.777), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.831 (+-0.038=0.792), grid train: 0.838, best classifier cross: 0.832 (+-0.037=0.795), best classifier:
	SVM - poly           - Stats: Default params cross: 0.815 (+-0.034=0.781), grid train: 0.826, best classifier cross: 0.822 (+-0.037=0.785), best classifier:
	NB                   - Stats: Default params cross: 0.782 (+-0.024=0.758), grid train: 0.791, best classifier cross: 0.782 (+-0.024=0.758), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.824 (+-0.05=0.774), grid train: 0.929, best classifier cross: 0.825 (+-0.047=0.778), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.842, best classifier cross: 0.822 (+-0.034=0.788), best classifier:

### 21. Try readding Fare with Log
	Logistic - liblinear - Stats: Default params cross: 0.798 (+-0.021=0.777), grid train: 0.804, best classifier cross: 0.798 (+-0.021=0.777), best classifier:
	KNN - 14             - Stats: Default params cross: 0.816 (+-0.033=0.783), grid train: 0.833, best classifier cross: 0.816 (+-0.033=0.783), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.828 (+-0.037=0.791), grid train: 0.838, best classifier cross: 0.828 (+-0.037=0.791), best classifier:
	SVM - poly           - Stats: Default params cross: 0.807 (+-0.03=0.777), grid train: 0.825, best classifier cross: 0.819 (+-0.037=0.782), best classifier:
	NB                   - Stats: Default params cross: 0.764 (+-0.029=0.735), grid train: 0.767, best classifier cross: 0.764 (+-0.029=0.735), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.824 (+-0.053=0.771), grid train: 0.926, best classifier cross: 0.826 (+-0.053=0.773), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.846, best classifier cross: 0.815 (+-0.027=0.788), best classifier:
Removing fare again - log is not better than without log, and Fare doesn't seem to help, makes it slightly worse

### 22. Adding some more hypermarams for grid searches of specific classifiers
	Logistic - liblinear - Stats: Default params cross: 0.8 (+-0.019=0.781), grid train: 0.811, best classifier cross: 0.801 (+-0.02=0.782), best classifier:
	KNN - 14             - Stats: Default params cross: 0.801 (+-0.037=0.764), grid train: 0.853, best classifier cross: 0.823 (+-0.051=0.772), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.824 (+-0.034=0.789), grid train: 0.836, best classifier cross: 0.827 (+-0.037=0.79), best classifier:
	SVM - poly           - Stats: Default params cross: 0.814 (+-0.034=0.78), grid train: 0.825, best classifier cross: 0.816 (+-0.036=0.78), best classifier:
	NB                   - Stats: Default params cross: 0.759 (+-0.032=0.727), grid train: 0.763, best classifier cross: 0.759 (+-0.032=0.727), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.831 (+-0.046=0.784), grid train: 0.901, best classifier cross: 0.823 (+-0.045=0.777), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.848, best classifier cross: 0.826 (+-0.038=0.789), best classifier:

### 23. Adding shuffling before fitting
	Logistic - liblinear - Stats: Default params cross: 0.804 (+-0.072=0.731), grid train: 0.811, best classifier cross: 0.807 (+-0.073=0.734), best classifier:
	KNN - 14             - Stats: Default params cross: 0.794 (+-0.051=0.743), grid train: 0.854, best classifier cross: 0.814 (+-0.038=0.776), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.823 (+-0.044=0.779), grid train: 0.835, best classifier cross: 0.828 (+-0.046=0.783), best classifier:
	SVM - poly           - Stats: Default params cross: 0.816 (+-0.047=0.769), grid train: 0.825, best classifier cross: 0.819 (+-0.047=0.772), best classifier:
	NB                   - Stats: Default params cross: 0.763 (+-0.039=0.724), grid train: 0.763, best classifier cross: 0.763 (+-0.039=0.724), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.826 (+-0.028=0.798), grid train: 0.9, best classifier cross: 0.824 (+-0.037=0.787), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.844, best classifier cross: 0.821 (+-0.049=0.771), best classifier:
Conclusion: doesn't seem to help, removing	

### 24. Adding XGBoost
	Logistic - liblinear - Stats: Default params cross: 0.8 (+-0.019=0.781), grid train: 0.811, best classifier cross: 0.801 (+-0.02=0.782), best classifier:
	KNN - 14             - Stats: Default params cross: 0.801 (+-0.037=0.764), grid train: 0.853, best classifier cross: 0.823 (+-0.051=0.772), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.824 (+-0.034=0.789), grid train: 0.836, best classifier cross: 0.827 (+-0.037=0.79), best classifier:
	SVM - poly           - Stats: Default params cross: 0.814 (+-0.034=0.78), grid train: 0.825, best classifier cross: 0.816 (+-0.036=0.78), best classifier:
	NB                   - Stats: Default params cross: 0.759 (+-0.032=0.727), grid train: 0.763, best classifier cross: 0.759 (+-0.032=0.727), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.827 (+-0.048=0.779), grid train: 0.9, best classifier cross: 0.823 (+-0.047=0.776), best classifier:
	XGB                  - Stats: Default params cross: 0.842 (+-0.033=0.808), grid train: 0.862, best classifier cross: 0.842 (+-0.033=0.808), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.85, best classifier cross: 0.831 (+-0.04=0.791), best classifier:
Conclusion: XGBoost gives good result (but maybe overfitting), using it gives better result also (not sure if overfitting)
On the website, adding XGBoost didn't improve the result of the voting, and having it on it's own actually made it worse

### 25. Adding running with different options - like not to do hyperparameter optimization every time
	Logistic - liblinear - Stats: Default params cross: 0.8 (+-0.019=0.781), grid train: 0.81, best classifier cross: 0.8 (+-0.019=0.781), best classifier:
	KNN - 14             - Stats: Default params cross: 0.801 (+-0.037=0.764), grid train: 0.824, best classifier cross: 0.801 (+-0.037=0.764), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.824 (+-0.034=0.789), grid train: 0.838, best classifier cross: 0.824 (+-0.034=0.789), best classifier:
	SVM - poly           - Stats: Default params cross: 0.814 (+-0.034=0.78), grid train: 0.828, best classifier cross: 0.814 (+-0.034=0.78), best classifier:
	NB                   - Stats: Default params cross: 0.759 (+-0.032=0.727), grid train: 0.763, best classifier cross: 0.759 (+-0.032=0.727), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.827 (+-0.042=0.785), grid train: 0.902, best classifier cross: 0.822 (+-0.041=0.78), best classifier:
	XGB                  - Stats: Default params cross: 0.842 (+-0.033=0.808), grid train: 0.862, best classifier cross: 0.842 (+-0.033=0.808), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.84, best classifier cross: 0.826 (+-0.037=0.789), best classifier:
	
### 26. Adding Deck feature learning it from Cabin
	Logistic - liblinear - Stats: Default params cross: 0.809 (+-0.027=0.782), grid train: 0.82, best classifier cross: 0.809 (+-0.027=0.782), best classifier:
	KNN - 14             - Stats: Default params cross: 0.81 (+-0.039=0.771), grid train: 0.825, best classifier cross: 0.81 (+-0.039=0.771), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.823 (+-0.033=0.79), grid train: 0.835, best classifier cross: 0.823 (+-0.033=0.79), best classifier:
	SVM - poly           - Stats: Default params cross: 0.816 (+-0.039=0.777), grid train: 0.838, best classifier cross: 0.816 (+-0.039=0.777), best classifier:
	NB                   - Stats: Default params cross: 0.754 (+-0.044=0.71), grid train: 0.765, best classifier cross: 0.754 (+-0.044=0.71), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.825 (+-0.04=0.785), grid train: 0.905, best classifier cross: 0.826 (+-0.042=0.784), best classifier:
	XGB                  - Stats: Default params cross: 0.84 (+-0.036=0.804), grid train: 0.865, best classifier cross: 0.84 (+-0.036=0.804), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.844, best classifier cross: 0.822 (+-0.034=0.788), best classifier:
Conclusion: doesn't seem to make a difference to have it or not to have it.  Leaving for now, perhaps will remove later.

### 27. Replacing unknown age based on median and not mean
	Logistic - liblinear - Stats: Default params cross: 0.809 (+-0.027=0.782), grid train: 0.822, best classifier cross: 0.809 (+-0.027=0.782), best classifier:
	KNN - 14             - Stats: Default params cross: 0.813 (+-0.04=0.773), grid train: 0.826, best classifier cross: 0.813 (+-0.04=0.773), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.823 (+-0.033=0.79), grid train: 0.835, best classifier cross: 0.823 (+-0.033=0.79), best classifier:
	SVM - poly           - Stats: Default params cross: 0.809 (+-0.041=0.768), grid train: 0.838, best classifier cross: 0.809 (+-0.041=0.768), best classifier:
	NB                   - Stats: Default params cross: 0.753 (+-0.049=0.705), grid train: 0.763, best classifier cross: 0.753 (+-0.049=0.705), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.818 (+-0.041=0.778), grid train: 0.897, best classifier cross: 0.816 (+-0.039=0.777), best classifier:
	XGB                  - Stats: Default params cross: 0.833 (+-0.033=0.8), grid train: 0.863, best classifier cross: 0.833 (+-0.033=0.8), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.843, best classifier cross: 0.817 (+-0.036=0.782), best classifier:
Conclusion: doesn't seem to make much of a difference, but makes sense.  Perhaps reconsider later.	

### 28. Adding binned Fare into 13 categories
	Logistic - liblinear - Stats: Default params cross: 0.808 (+-0.029=0.779), grid train: 0.818, best classifier cross: 0.808 (+-0.029=0.779), best classifier:
	KNN - 14             - Stats: Default params cross: 0.818 (+-0.038=0.781), grid train: 0.837, best classifier cross: 0.818 (+-0.038=0.781), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.823 (+-0.033=0.789), grid train: 0.838, best classifier cross: 0.823 (+-0.033=0.789), best classifier:
	SVM - poly           - Stats: Default params cross: 0.831 (+-0.042=0.788), grid train: 0.847, best classifier cross: 0.831 (+-0.042=0.788), best classifier:
	NB                   - Stats: Default params cross: 0.752 (+-0.036=0.716), grid train: 0.753, best classifier cross: 0.752 (+-0.036=0.716), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.827 (+-0.039=0.788), grid train: 0.915, best classifier cross: 0.822 (+-0.039=0.782), best classifier:
	XGB                  - Stats: Default params cross: 0.835 (+-0.04=0.795), grid train: 0.864, best classifier cross: 0.835 (+-0.04=0.795), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.851, best classifier cross: 0.825 (+-0.036=0.789), best classifier:
Conclusion: seems to help a bit, leaving for now.

### 29. Changing age into bins (9)
	Logistic - liblinear - Stats: Default params cross: 0.792 (+-0.033=0.759), grid train: 0.808, best classifier cross: 0.792 (+-0.033=0.759), best classifier:
	KNN - 14             - Stats: Default params cross: 0.813 (+-0.046=0.767), grid train: 0.826, best classifier cross: 0.813 (+-0.046=0.767), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.802 (+-0.025=0.778), grid train: 0.825, best classifier cross: 0.802 (+-0.025=0.778), best classifier:
	SVM - poly           - Stats: Default params cross: 0.822 (+-0.042=0.78), grid train: 0.842, best classifier cross: 0.822 (+-0.042=0.78), best classifier:
	NB                   - Stats: Default params cross: 0.751 (+-0.039=0.712), grid train: 0.752, best classifier cross: 0.751 (+-0.039=0.712), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.825 (+-0.044=0.781), grid train: 0.906, best classifier cross: 0.819 (+-0.042=0.778), best classifier:
	XGB                  - Stats: Default params cross: 0.83 (+-0.036=0.794), grid train: 0.864, best classifier cross: 0.83 (+-0.036=0.794), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.846, best classifier cross: 0.814 (+-0.038=0.776), best classifier:
Conclusion: looks like makes things worse, reverting for now

### 30. Changing number of children / spouces / parents into 1 category with 3 groups: alone, small family, large family
	Logistic - liblinear - Stats: Default params cross: 0.823 (+-0.025=0.797), grid train: 0.832, best classifier cross: 0.823 (+-0.025=0.797), best classifier:
	KNN - 14             - Stats: Default params cross: 0.815 (+-0.027=0.788), grid train: 0.834, best classifier cross: 0.815 (+-0.027=0.788), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.834 (+-0.03=0.804), grid train: 0.847, best classifier cross: 0.834 (+-0.03=0.804), best classifier:
	SVM - poly           - Stats: Default params cross: 0.844 (+-0.043=0.801), grid train: 0.854, best classifier cross: 0.844 (+-0.043=0.801), best classifier:
	NB                   - Stats: Default params cross: 0.754 (+-0.046=0.708), grid train: 0.756, best classifier cross: 0.754 (+-0.046=0.708), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.836 (+-0.039=0.797), grid train: 0.912, best classifier cross: 0.841 (+-0.041=0.799), best classifier:
	XGB                  - Stats: Default params cross: 0.844 (+-0.042=0.802), grid train: 0.868, best classifier cross: 0.844 (+-0.042=0.802), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.863, best classifier cross: 0.834 (+-0.033=0.801), best classifier:
Conclusion: helps, leaving

### 31. Adding ticket frequency - similar to family size, but maybe traveling in groups that are not family
	Line 115: Logistic - liblinear - Stats: Default params cross: 0.824 (+-0.025=0.798), grid train: 0.832, best classifier cross: 0.824 (+-0.025=0.798), best classifier:
	Line 125: KNN - 14             - Stats: Default params cross: 0.817 (+-0.031=0.786), grid train: 0.835, best classifier cross: 0.817 (+-0.031=0.786), best classifier:
	Line 133: SVM - rbf            - Stats: Default params cross: 0.835 (+-0.035=0.8), grid train: 0.851, best classifier cross: 0.835 (+-0.035=0.8), best classifier:
	Line 142: SVM - poly           - Stats: Default params cross: 0.843 (+-0.044=0.799), grid train: 0.859, best classifier cross: 0.843 (+-0.044=0.799), best classifier:
	Line 153: NB                   - Stats: Default params cross: 0.757 (+-0.045=0.712), grid train: 0.758, best classifier cross: 0.757 (+-0.045=0.712), best classifier:
	Line 159: RandomForest - 9     - Stats: Default params cross: 0.837 (+-0.04=0.798), grid train: 0.91, best classifier cross: 0.837 (+-0.038=0.799), best classifier:
	Line 171: XGB                  - Stats: Default params cross: 0.844 (+-0.034=0.81), grid train: 0.871, best classifier cross: 0.844 (+-0.034=0.81), best classifier:
	Line 551: FINAL                                        - Stats: Best classifiers + Voting train: 0.861, best classifier cross: 0.833 (+-0.03=0.802), best classifier:
Conclusion: doesn't make a change, (XGB seems to improve) but leaving for now

### 32. Adding Lady married feature based on Title Mrs
	Logistic - liblinear - Stats: Default params cross: 0.825 (+-0.031=0.793), grid train: 0.829, best classifier cross: 0.825 (+-0.031=0.793), best classifier:
	KNN - 14             - Stats: Default params cross: 0.822 (+-0.036=0.786), grid train: 0.842, best classifier cross: 0.822 (+-0.036=0.786), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.837 (+-0.03=0.807), grid train: 0.85, best classifier cross: 0.837 (+-0.03=0.807), best classifier:
	SVM - poly           - Stats: Default params cross: 0.838 (+-0.041=0.797), grid train: 0.863, best classifier cross: 0.838 (+-0.041=0.797), best classifier:
	NB                   - Stats: Default params cross: 0.763 (+-0.037=0.726), grid train: 0.767, best classifier cross: 0.763 (+-0.037=0.726), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.842 (+-0.045=0.797), grid train: 0.915, best classifier cross: 0.837 (+-0.046=0.792), best classifier:
	XGB                  - Stats: Default params cross: 0.838 (+-0.034=0.805), grid train: 0.874, best classifier cross: 0.838 (+-0.034=0.805), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.862, best classifier cross: 0.842 (+-0.035=0.806), best classifier:
Conclusion: seems to improve, leaving. On site gives ~76 result (worse than before adding all these features)

### 33. Added Survival rates based on family (based on last name) and ticket number
	Logistic - liblinear - Stats: Default params cross: 0.984 (+-0.01=0.974), grid train: 0.988, best classifier cross: 0.984 (+-0.01=0.974), best classifier:
	KNN - 14             - Stats: Default params cross: 0.934 (+-0.026=0.908), grid train: 0.952, best classifier cross: 0.934 (+-0.026=0.908), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.966 (+-0.019=0.948), grid train: 0.988, best classifier cross: 0.966 (+-0.019=0.948), best classifier:
	SVM - poly           - Stats: Default params cross: 0.974 (+-0.022=0.952), grid train: 0.984, best classifier cross: 0.974 (+-0.022=0.952), best classifier:
	NB                   - Stats: Default params cross: 0.943 (+-0.025=0.918), grid train: 0.947, best classifier cross: 0.943 (+-0.025=0.918), best classifier:
	RandomForest - 9     - Stats: Default params cross: 0.988 (+-0.009=0.978), grid train: 0.999, best classifier cross: 0.987 (+-0.01=0.977), best classifier:
	XGB                  - Stats: Default params cross: 0.984 (+-0.013=0.972), grid train: 0.996, best classifier cross: 0.984 (+-0.013=0.972), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.992, best classifier cross: 0.984 (+-0.012=0.972), best classifier:
Conclusion: Amazing results, but it seems that it's doing overfitting, the results are not real since the result is already coded in the inputs.  
Indeed, doing in the website gives a much worse result ~71

### 34. Put RandomForest outside of grid to get values, want to look at importances of features
	Feature importances (RandomForest Specific):
	Family/ticket survival known      0.000000
	Deck_FG                           0.000269
	Embarked_Q                        0.001114
	Deck_AC                           0.002082
	Deck_BT                           0.003505
	Embarked_S                        0.003576
	pclass_2                          0.005516
	Deck_DE                           0.009190
	Large family                      0.010095
	Small family                      0.017605
	Ticket_Frequency                  0.020562
	pclass_1                          0.023262
	Age                               0.030158
	Fare                              0.034578
	Lady married                      0.038785
	Male                              0.163035
	Known family/ticket survived %    0.636666
based on this, removed the following features for next run: 
 'Family/ticket survival known','Deck_FG','Embarked_Q','Deck_AC','Deck_BT','Embarked_S'
 Also changed the default Random Forest to be 5 max levels, and not 9:
	Logistic - liblinear - Stats: Default params cross: 0.985 (+-0.01=0.975), grid train: 0.989, best classifier cross: 0.985 (+-0.01=0.975), best classifier:
	KNN - 14             - Stats: Default params cross: 0.956 (+-0.021=0.935), grid train: 0.964, best classifier cross: 0.956 (+-0.021=0.935), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.985 (+-0.01=0.975), grid train: 0.993, best classifier cross: 0.985 (+-0.01=0.975), best classifier:
	SVM - poly           - Stats: Default params cross: 0.981 (+-0.012=0.969), grid train: 0.989, best classifier cross: 0.981 (+-0.012=0.969), best classifier:
	NB                   - Stats: Default params cross: 0.948 (+-0.02=0.928), grid train: 0.951, best classifier cross: 0.948 (+-0.02=0.928), best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.984 (+-0.013=0.972), grid train: 0.997, best classifier cross: 0.986 (+-0.011=0.975), best classifier:
	RandomForest Specific - Stats: Default params cross: grid train: 0.992, best classifier cross: 0.982 (+-0.014=0.968)
	RandomForest Specific - Stats: Default params cross: grid train: 0.992, best classifier cross: 0.983 (+-0.014=0.969)
	XGB                  - Stats: Default params cross: 0.987 (+-0.013=0.973), grid train: 0.996, best classifier cross: 0.987 (+-0.013=0.973), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.992, best classifier cross: 0.988 (+-0.009=0.978), best classifier:

### 35. Change survival rate not to take current training survival result into account
	Logistic - liblinear - Stats: Default params cross: 0.832 (+-0.025=0.807), grid train: 0.833, best classifier cross: 0.832 (+-0.025=0.807), best classifier:
	KNN - 14             - Stats: Default params cross: 0.798 (+-0.023=0.775), grid train: 0.833, best classifier cross: 0.798 (+-0.023=0.775), best classifier:
	SVM - rbf            - Stats: Default params cross: 0.836 (+-0.042=0.794), grid train: 0.871, best classifier cross: 0.836 (+-0.042=0.794), best classifier:
	SVM - poly           - Stats: Default params cross: 0.827 (+-0.038=0.789), grid train: 0.875, best classifier cross: 0.827 (+-0.038=0.789), best classifier:
	NB                   - Stats: Default params cross: 0.76 (+-0.044=0.716), grid train: 0.761, best classifier cross: 0.76 (+-0.044=0.716), best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.829 (+-0.019=0.81), grid train: 0.863, best classifier cross: 0.834 (+-0.028=0.806), best classifier:
	RandomForest Specific - Stats: Default params cross: grid train: 0.859, best classifier cross: 0.836 (+-0.026=0.81)
	RandomForest Specific - Stats: Default params cross: grid train: 0.877, best classifier cross: 0.834 (+-0.029=0.805)
	XGB                  - Stats: Default params cross: 0.844 (+-0.034=0.81), grid train: 0.888, best classifier cross: 0.844 (+-0.034=0.81), best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.869, best classifier cross: 0.833 (+-0.026=0.807), best classifier:
Conclusion: hopefully not overfitting this time, results went down drastically, similar to before.  Got much better result on the website: ~0.89 with voting.

### 36. Cleanup 
- Remove Random Forest Leader - didn't give better results, overfitting for model I was learning from 
- Make 2 RandomForests to behave the same way, leave explicit RandomForest to learn about important and not important features
- Make regular RandomForest do many more trees
- Cleanup output files to contain only what I want: RandomForest, XGB, Voting
- Split up regular RandomForest into 4 and 5 depths, possibly since I have slightly less features, depth 4 (and maybe even 3) might do better than 5:
- Measure time what everything takes so can decide if continue doing it
- Not to return Grid from Grid, but best classifier

Results:
	Logistic - liblinear - Stats: Default params cross: 0.832 (+-0.025=0.807), grid train: 0.833, best classifier cross: 0.832 (+-0.025=0.807), time took: 16 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.798 (+-0.023=0.775), grid train: 0.833, best classifier cross: 0.798 (+-0.023=0.775), time took: 2 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.836 (+-0.042=0.794), grid train: 0.871, best classifier cross: 0.836 (+-0.042=0.794), time took: 2 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.827 (+-0.038=0.789), grid train: 0.875, best classifier cross: 0.827 (+-0.038=0.789), time took: 2 sec, best classifier:
	NB                   - Stats: Default params cross: 0.76 (+-0.044=0.716), grid train: 0.761, best classifier cross: 0.76 (+-0.044=0.716), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.841 (+-0.023=0.818), grid train: 0.869, best classifier cross: 0.834 (+-0.033=0.801), time took: 28 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.832 (+-0.025=0.806), grid train: 0.853, best classifier cross: 0.834 (+-0.026=0.808), time took: 30 sec, best classifier:
	RandomForest - 3     - Stats: Default params cross: 0.821 (+-0.027=0.793), grid train: 0.842, best classifier cross: 0.821 (+-0.031=0.789), time took: 31 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.871, best classifier cross: 0.837 (+-0.028=0.809), time took: 14 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.844 (+-0.034=0.81), grid train: 0.888, best classifier cross: 0.844 (+-0.034=0.81), time took: 2 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.869, best classifier cross: 0.832 (+-0.028=0.803), time took: 41 sec, best classifier:
Conclusion: now ready to remove some features

### 37. Added reference categories 1. To see which ones to delete 2. To see if deleting gives worse results
	Logistic - liblinear - Stats: Default params cross: 0.829 (+-0.027=0.803), grid train: 0.833, best classifier cross: 0.829 (+-0.027=0.803), time took: 15 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.791 (+-0.034=0.758), grid train: 0.819, best classifier cross: 0.791 (+-0.034=0.758), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.837 (+-0.039=0.798), grid train: 0.866, best classifier cross: 0.837 (+-0.039=0.798), time took: 2 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.829 (+-0.036=0.794), grid train: 0.875, best classifier cross: 0.829 (+-0.036=0.794), time took: 1 sec, best classifier:
	NB                   - Stats: Default params cross: 0.758 (+-0.047=0.711), grid train: 0.758, best classifier cross: 0.758 (+-0.047=0.711), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.843 (+-0.035=0.808), grid train: 0.873, best classifier cross: 0.845 (+-0.032=0.814), time took: 26 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.834 (+-0.035=0.799), grid train: 0.862, best classifier cross: 0.835 (+-0.035=0.8), time took: 25 sec, best classifier:
	RandomForest - 3     - Stats: Default params cross: 0.81 (+-0.036=0.774), grid train: 0.833, best classifier cross: 0.81 (+-0.035=0.775), time took: 28 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.872, best classifier cross: 0.842 (+-0.03=0.812), time took: 13 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.837 (+-0.034=0.803), grid train: 0.872, best classifier cross: 0.837 (+-0.034=0.803), time took: 2 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.865, best classifier cross: 0.836 (+-0.03=0.806), time took: 40 sec, best classifier:

Feature importances:
	Deck_FG                           0.001825
	Deck_AC                           0.005390
	Deck_BT                           0.006403
	Embarked_Q                        0.006742
	Embarked_C                        0.008113
	Family/ticket survival known      0.009555
	Embarked_S                        0.010505
	pclass_2                          0.011551
	Alone                             0.011666
	Deck_DE                           0.017524
	Large family                      0.018405
	Small family                      0.028471
	pclass_1                          0.035444
	Ticket_Frequency                  0.049215
	Deck_Other                        0.054742
	pclass_3                          0.060746
	Fare                              0.068090
	Lady married                      0.081713
	Age                               0.082346
	Known family/ticket survived %    0.123656
	Male                              0.307899

Conclusion: 
- For pclass remove pclass_2
- For family size - remove Alone as was before
- For Deck - only add deck DE and Deck_Other, remove rest
- Remove all the Embarked, don't seem to make a difference
It's not exactly consisent with notebook I learned different things from, but my model is different https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic

### 38. Removed reference features
Removed:
	'pclass_2',
	'Alone',
	'Deck_FG',
	'Deck_AC',
	'Deck_BT',
	'Embarked' - all embarked
Results:
	Logistic - liblinear - Stats: Default params cross: 0.834 (+-0.025=0.809), grid train: 0.836, best classifier cross: 0.834 (+-0.025=0.809), time took: 14 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.836 (+-0.031=0.805), grid train: 0.852, best classifier cross: 0.836 (+-0.031=0.805), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.851 (+-0.027=0.823), grid train: 0.87, best classifier cross: 0.851 (+-0.027=0.823), time took: 2 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.837 (+-0.027=0.81), grid train: 0.864, best classifier cross: 0.837 (+-0.027=0.81), time took: 1 sec, best classifier:
	NB                   - Stats: Default params cross: 0.762 (+-0.053=0.71), grid train: 0.763, best classifier cross: 0.762 (+-0.053=0.71), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.838 (+-0.03=0.809), grid train: 0.874, best classifier cross: 0.837 (+-0.025=0.812), time took: 27 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.832 (+-0.041=0.791), grid train: 0.854, best classifier cross: 0.83 (+-0.039=0.791), time took: 23 sec, best classifier:
	RandomForest - 3     - Stats: Default params cross: 0.814 (+-0.036=0.778), grid train: 0.831, best classifier cross: 0.815 (+-0.037=0.778), time took: 26 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.87, best classifier cross: 0.833 (+-0.027=0.806), time took: 14 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.847 (+-0.034=0.814), grid train: 0.888, best classifier cross: 0.847 (+-0.034=0.814), time took: 2 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.866, best classifier cross: 0.84 (+-0.024=0.816), time took: 40 sec, best classifier:
Conclusion: both voting, and XGB gave much better results.  Most classifiers did better, besides RandomForest, which is probably a good thing since it probably overfits 

### 39. Removing 'Family/ticket survival known' since it's both low coefficient, and the mean of survival already coded in 'Known family/ticket survived %'
	Logistic - liblinear - Stats: Default params cross: 0.833 (+-0.024=0.809), grid train: 0.835, best classifier cross: 0.833 (+-0.024=0.809), time took: 16 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.832 (+-0.032=0.8), grid train: 0.846, best classifier cross: 0.832 (+-0.032=0.8), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.849 (+-0.026=0.822), grid train: 0.869, best classifier cross: 0.849 (+-0.026=0.822), time took: 2 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.835 (+-0.037=0.798), grid train: 0.872, best classifier cross: 0.835 (+-0.037=0.798), time took: 1 sec, best classifier:
	NB                   - Stats: Default params cross: 0.762 (+-0.054=0.709), grid train: 0.769, best classifier cross: 0.762 (+-0.054=0.709), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.834 (+-0.027=0.807), grid train: 0.873, best classifier cross: 0.834 (+-0.027=0.807), time took: 28 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.828 (+-0.041=0.787), grid train: 0.86, best classifier cross: 0.828 (+-0.041=0.788), time took: 25 sec, best classifier:
	RandomForest - 3     - Stats: Default params cross: 0.813 (+-0.035=0.777), grid train: 0.831, best classifier cross: 0.816 (+-0.036=0.78), time took: 26 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.873, best classifier cross: 0.837 (+-0.025=0.812), time took: 13 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.847 (+-0.034=0.814), grid train: 0.888, best classifier cross: 0.847 (+-0.034=0.814), time took: 2 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.864, best classifier cross: 0.842 (+-0.031=0.811), time took: 44 sec, best classifier:
Conclusion: similar, but slightly worse results (especially in voting), leave it in

### 40. Doing 'Age' as binned category (that's what was done in kernel I learned from)
	Logistic - liblinear - Stats: Default params cross: 0.829 (+-0.023=0.806), grid train: 0.832, best classifier cross: 0.829 (+-0.023=0.806), time took: 14 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.831 (+-0.032=0.799), grid train: 0.854, best classifier cross: 0.831 (+-0.032=0.799), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.842 (+-0.029=0.813), grid train: 0.863, best classifier cross: 0.842 (+-0.029=0.813), time took: 2 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.844 (+-0.027=0.817), grid train: 0.866, best classifier cross: 0.844 (+-0.027=0.817), time took: 1 sec, best classifier:
	NB                   - Stats: Default params cross: 0.761 (+-0.051=0.711), grid train: 0.762, best classifier cross: 0.761 (+-0.051=0.711), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.836 (+-0.025=0.811), grid train: 0.871, best classifier cross: 0.837 (+-0.024=0.813), time took: 25 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.833 (+-0.04=0.793), grid train: 0.86, best classifier cross: 0.833 (+-0.038=0.795), time took: 27 sec, best classifier:
	RandomForest - 3     - Stats: Default params cross: 0.812 (+-0.034=0.778), grid train: 0.822, best classifier cross: 0.814 (+-0.038=0.776), time took: 25 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.871, best classifier cross: 0.842 (+-0.025=0.817), time took: 11 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.846 (+-0.031=0.816), grid train: 0.881, best classifier cross: 0.846 (+-0.031=0.816), time took: 1 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.863, best classifier cross: 0.838 (+-0.022=0.817), time took: 40 sec, best classifier:
Conclusion: seems very similar if Age is binned or not (btw, wasn't able to bin into 10, only 11 because of unique values).  
Since using not only RandomForests, leaving Age as a number

### 41. Try various options on website
	Logistic - liblinear - Stats: Default params cross: 0.834 (+-0.025=0.809), grid train: 0.836, best classifier cross: 0.834 (+-0.025=0.809), time took: 15 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.836 (+-0.031=0.805), grid train: 0.852, best classifier cross: 0.836 (+-0.031=0.805), time took: 2 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.851 (+-0.027=0.823), grid train: 0.87, best classifier cross: 0.851 (+-0.027=0.823), time took: 2 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.837 (+-0.027=0.81), grid train: 0.864, best classifier cross: 0.837 (+-0.027=0.81), time took: 1 sec, best classifier:
	NB                   - Stats: Default params cross: 0.762 (+-0.053=0.71), grid train: 0.763, best classifier cross: 0.762 (+-0.053=0.71), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.838 (+-0.027=0.812), grid train: 0.872, best classifier cross: 0.836 (+-0.025=0.811), time took: 24 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.826 (+-0.042=0.785), grid train: 0.856, best classifier cross: 0.825 (+-0.043=0.782), time took: 27 sec, best classifier:
	RandomForest - 3     - Stats: Default params cross: 0.814 (+-0.039=0.775), grid train: 0.829, best classifier cross: 0.815 (+-0.038=0.777), time took: 25 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.873, best classifier cross: 0.836 (+-0.032=0.804), time took: 11 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.847 (+-0.034=0.814), grid train: 0.888, best classifier cross: 0.847 (+-0.034=0.814), time took: 1 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.866, best classifier cross: 0.836 (+-0.026=0.81), time took: 35 sec, best classifier:
Conclusions: 
- RandomForest 4 is worse than 5, so leaving 4 for now, but completely removing 3
- Doing directly Random Forest is better than doing voting (perhaps because the specific features are more specific for RandomForest)
- Voting on website: ~.81, XGB: ~.77 (perhaps need to optimize), RandomForest 5: ~.82, RandomForest 4: ~.815

### 42. Making XGB slightly better parametrized, take SVM directly
	Logistic - liblinear - Stats: Default params cross: 0.834 (+-0.025=0.809), grid train: 0.836, best classifier cross: 0.834 (+-0.025=0.809), time took: 14 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.836 (+-0.031=0.805), grid train: 0.852, best classifier cross: 0.836 (+-0.031=0.805), time took: 2 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.851 (+-0.027=0.823), grid train: 0.87, best classifier cross: 0.851 (+-0.027=0.823), time took: 1 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.837 (+-0.027=0.81), grid train: 0.864, best classifier cross: 0.837 (+-0.027=0.81), time took: 1 sec, best classifier:
	NB                   - Stats: Default params cross: 0.762 (+-0.053=0.71), grid train: 0.763, best classifier cross: 0.762 (+-0.053=0.71), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.842 (+-0.025=0.816), grid train: 0.871, best classifier cross: 0.835 (+-0.026=0.809), time took: 25 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.831 (+-0.042=0.789), grid train: 0.856, best classifier cross: 0.832 (+-0.041=0.791), time took: 23 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.871, best classifier cross: 0.838 (+-0.028=0.811), time took: 11 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.827 (+-0.032=0.795), grid train: 0.963, best classifier cross: 0.827 (+-0.032=0.795), time took: 15 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.895, best classifier cross: 0.844 (+-0.035=0.81), time took: 32 sec, best classifier:
Conclusions: seems that XGB got worse (perhaps because it's overfitting, since max_depth was changed from 3 to 5), try 4

### 43. Try SVM directly, revert some XGB changes
Revert XGB to be depth 4 (default was 3) and not 5 as last time
	Logistic - liblinear - Stats: Default params cross: 0.834 (+-0.025=0.809), grid train: 0.836, best classifier cross: 0.834 (+-0.025=0.809), time took: 15 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.836 (+-0.031=0.805), grid train: 0.852, best classifier cross: 0.836 (+-0.031=0.805), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.851 (+-0.027=0.823), grid train: 0.87, best classifier cross: 0.851 (+-0.027=0.823), time took: 2 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.837 (+-0.027=0.81), grid train: 0.864, best classifier cross: 0.837 (+-0.027=0.81), time took: 1 sec, best classifier:
	NB                   - Stats: Default params cross: 0.762 (+-0.053=0.71), grid train: 0.763, best classifier cross: 0.762 (+-0.053=0.71), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.836 (+-0.024=0.812), grid train: 0.872, best classifier cross: 0.84 (+-0.028=0.811), time took: 24 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.828 (+-0.043=0.786), grid train: 0.855, best classifier cross: 0.826 (+-0.04=0.786), time took: 24 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.874, best classifier cross: 0.835 (+-0.02=0.815), time took: 10 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.842 (+-0.035=0.807), grid train: 0.955, best classifier cross: 0.842 (+-0.035=0.807), time took: 11 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.896, best classifier cross: 0.841 (+-0.036=0.805), time took: 30 sec, best classifier:
Conclusions: XGB is much better now, so better 4 than 5, but not sure if 3 or 4 depth is better. Reverting to default - seems to overfit like this
SVM - rbf on the website gave ~80.3 score, which is not better than RandomForest, but very good.  
Removing NB, and RandomForest (4) from voting, since seem to make things worse.  Not sure if also to remove XGB, since with a high score, it could be taking votes from let's say RandomForest

### 44. Removed NB, and RandomForest (4) from voting. Left XGB for now
	Logistic - liblinear - Stats: Default params cross: 0.834 (+-0.025=0.809), grid train: 0.836, best classifier cross: 0.834 (+-0.025=0.809), time took: 16 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.836 (+-0.031=0.805), grid train: 0.852, best classifier cross: 0.836 (+-0.031=0.805), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.851 (+-0.027=0.823), grid train: 0.87, best classifier cross: 0.851 (+-0.027=0.823), time took: 3 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.837 (+-0.027=0.81), grid train: 0.864, best classifier cross: 0.837 (+-0.027=0.81), time took: 1 sec, best classifier:
	NB                   - Stats: Default params cross: 0.762 (+-0.053=0.71), grid train: 0.763, best classifier cross: 0.762 (+-0.053=0.71), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.833 (+-0.023=0.81), grid train: 0.872, best classifier cross: 0.841 (+-0.022=0.819), time took: 26 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.831 (+-0.043=0.788), grid train: 0.86, best classifier cross: 0.833 (+-0.043=0.79), time took: 28 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.873, best classifier cross: 0.838 (+-0.026=0.812), time took: 10 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.844 (+-0.025=0.819), grid train: 0.937, best classifier cross: 0.844 (+-0.025=0.819), time took: 9 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.882, best classifier cross: 0.846 (+-0.032=0.815), time took: 21 sec, best classifier:
Conclusion: Not clear if better with or without Logistic, KNN, XGB, don't have enough tries on the website to know
On the site, the voting is ~.80 down from ~.81, try to remove XGB and return others

### 45. Returned low performance stuff, removed XGB altogether
	Line 236: Logistic - liblinear - Stats: Default params cross: 0.834 (+-0.025=0.809), grid train: 0.836, best classifier cross: 0.834 (+-0.025=0.809), time took: 14 sec, best classifier:
	Line 248: KNN - 14             - Stats: Default params cross: 0.836 (+-0.031=0.805), grid train: 0.852, best classifier cross: 0.836 (+-0.031=0.805), time took: 1 sec, best classifier:
	Line 256: SVM - rbf            - Stats: Default params cross: 0.851 (+-0.027=0.823), grid train: 0.87, best classifier cross: 0.851 (+-0.027=0.823), time took: 1 sec, best classifier:
	Line 265: SVM - poly           - Stats: Default params cross: 0.837 (+-0.027=0.81), grid train: 0.864, best classifier cross: 0.837 (+-0.027=0.81), time took: 1 sec, best classifier:
	Line 276: NB                   - Stats: Default params cross: 0.762 (+-0.053=0.71), grid train: 0.763, best classifier cross: 0.762 (+-0.053=0.71), time took: 0 sec, best classifier:
	Line 282: RandomForest - 5     - Stats: Default params cross: 0.838 (+-0.022=0.816), grid train: 0.873, best classifier cross: 0.84 (+-0.028=0.812), time took: 24 sec, best classifier:
	Line 294: RandomForest - 4     - Stats: Default params cross: 0.822 (+-0.036=0.786), grid train: 0.855, best classifier cross: 0.831 (+-0.037=0.794), time took: 24 sec, best classifier:
	Line 302: RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.873, best classifier cross: 0.837 (+-0.025=0.813), time took: 10 sec = 0 min, importances:
	Line 321: XGB                  - Stats: Default params cross: 0.844 (+-0.025=0.819), grid train: 0.937, best classifier cross: 0.844 (+-0.025=0.819), time took: 9 sec, best classifier:
	Line 351: FINAL                                        - Stats: Best classifiers + Voting train: 0.859, best classifier cross: 0.838 (+-0.032=0.807), time took: 25 sec, best classifier:
Conclusion: The scores went down, but perhaps it's due to less overfitting.
Score of voting on the site - back to ~.81, so it doesn't matter much if XGB is inside or not

### 46. Doing Random Forest with exact parameters as in the kernel I'm learning from
	Logistic - liblinear - Stats: Default params cross: 0.834 (+-0.025=0.809), grid train: 0.836, best classifier cross: 0.834 (+-0.025=0.809), time took: 14 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.836 (+-0.031=0.805), grid train: 0.852, best classifier cross: 0.836 (+-0.031=0.805), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.851 (+-0.027=0.823), grid train: 0.87, best classifier cross: 0.851 (+-0.027=0.823), time took: 1 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.837 (+-0.027=0.81), grid train: 0.864, best classifier cross: 0.837 (+-0.027=0.81), time took: 1 sec, best classifier:
	NB                   - Stats: Default params cross: 0.762 (+-0.053=0.71), grid train: 0.763, best classifier cross: 0.762 (+-0.053=0.71), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.836 (+-0.03=0.806), grid train: 0.877, best classifier cross: 0.833 (+-0.022=0.811), time took: 22 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.826 (+-0.041=0.785), grid train: 0.859, best classifier cross: 0.831 (+-0.038=0.792), time took: 22 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.864, best classifier cross: 0.837 (+-0.029=0.809), time took: 15 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.844 (+-0.025=0.819), grid train: 0.937, best classifier cross: 0.844 (+-0.025=0.819), time took: 9 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.883, best classifier cross: 0.841 (+-0.034=0.807), time took: 35 sec, best classifier:
Conclusion: locally pretty much the same, on website: no different than default params: ~.82

### 47. Doing Random Forest with exact parameters as in the kernel I'm learning from (Leader kernel - overfitting)
	Line 236: Logistic - liblinear - Stats: Default params cross: 0.834 (+-0.025=0.809), grid train: 0.836, best classifier cross: 0.834 (+-0.025=0.809), time took: 13 sec, best classifier:
	Line 248: KNN - 14             - Stats: Default params cross: 0.836 (+-0.031=0.805), grid train: 0.852, best classifier cross: 0.836 (+-0.031=0.805), time took: 1 sec, best classifier:
	Line 256: SVM - rbf            - Stats: Default params cross: 0.851 (+-0.027=0.823), grid train: 0.87, best classifier cross: 0.851 (+-0.027=0.823), time took: 1 sec, best classifier:
	Line 265: SVM - poly           - Stats: Default params cross: 0.837 (+-0.027=0.81), grid train: 0.864, best classifier cross: 0.837 (+-0.027=0.81), time took: 1 sec, best classifier:
	Line 276: NB                   - Stats: Default params cross: 0.762 (+-0.053=0.71), grid train: 0.763, best classifier cross: 0.762 (+-0.053=0.71), time took: 0 sec, best classifier:
	Line 282: RandomForest - 5     - Stats: Default params cross: 0.836 (+-0.026=0.81), grid train: 0.873, best classifier cross: 0.838 (+-0.028=0.81), time took: 22 sec, best classifier:
	Line 294: RandomForest - 4     - Stats: Default params cross: 0.831 (+-0.04=0.791), grid train: 0.859, best classifier cross: 0.832 (+-0.037=0.795), time took: 24 sec, best classifier:
	Line 302: RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.868, best classifier cross: 0.84 (+-0.029=0.811), time took: 28 sec = 0 min, importances:
	Line 321: XGB                  - Stats: Default params cross: 0.844 (+-0.025=0.819), grid train: 0.937, best classifier cross: 0.844 (+-0.025=0.819), time took: 10 sec, best classifier:
	Line 351: FINAL                                        - Stats: Best classifiers + Voting train: 0.883, best classifier cross: 0.843 (+-0.033=0.81), time took: 27 sec, best classifier:
Conclusion: Leader is worse, returning to default params

### 48. Final benchmark results, making a checkpoint:
	Logistic - liblinear - Stats: Default params cross: 0.834 (+-0.025=0.809), grid train: 0.836, best classifier cross: 0.834 (+-0.025=0.809), time took: 13 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.836 (+-0.031=0.805), grid train: 0.852, best classifier cross: 0.836 (+-0.031=0.805), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.851 (+-0.027=0.823), grid train: 0.87, best classifier cross: 0.851 (+-0.027=0.823), time took: 1 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.837 (+-0.027=0.81), grid train: 0.864, best classifier cross: 0.837 (+-0.027=0.81), time took: 1 sec, best classifier:
	NB                   - Stats: Default params cross: 0.762 (+-0.053=0.71), grid train: 0.763, best classifier cross: 0.762 (+-0.053=0.71), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.838 (+-0.027=0.812), grid train: 0.875, best classifier cross: 0.842 (+-0.028=0.813), time took: 22 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.824 (+-0.042=0.782), grid train: 0.856, best classifier cross: 0.832 (+-0.041=0.791), time took: 24 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.873, best classifier cross: 0.84 (+-0.026=0.813), time took: 11 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.844 (+-0.025=0.819), grid train: 0.937, best classifier cross: 0.844 (+-0.025=0.819), time took: 10 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.883, best classifier cross: 0.841 (+-0.034=0.807), time took: 30 sec, best classifier:
On website: RandomForest (5) ~.82, Voting ~.81, XGB - ~.77

### 49. Adding a title feature: Mr, Mrs, Miss, Master, (everything else was put into these 4 based on gender, age and assumed marital status) 
	Logistic - liblinear - Stats: Default params cross: 0.844 (+-0.029=0.815), grid train: 0.85, best classifier cross: 0.844 (+-0.029=0.815), time took: 21 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.831 (+-0.033=0.798), grid train: 0.846, best classifier cross: 0.831 (+-0.033=0.798), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.834 (+-0.038=0.796), grid train: 0.873, best classifier cross: 0.834 (+-0.038=0.796), time took: 2 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.844 (+-0.036=0.808), grid train: 0.881, best classifier cross: 0.844 (+-0.036=0.808), time took: 2 sec, best classifier:
	NB                   - Stats: Default params cross: 0.793 (+-0.046=0.746), grid train: 0.798, best classifier cross: 0.793 (+-0.046=0.746), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.842 (+-0.033=0.809), grid train: 0.859, best classifier cross: 0.844 (+-0.031=0.813), time took: 34 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.841 (+-0.03=0.811), grid train: 0.848, best classifier cross: 0.842 (+-0.03=0.812), time took: 31 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.856, best classifier cross: 0.844 (+-0.037=0.807), time took: 14 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.826 (+-0.031=0.795), grid train: 0.948, best classifier cross: 0.826 (+-0.031=0.795), time took: 20 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.886, best classifier cross: 0.846 (+-0.033=0.813), time took: 37 sec, best classifier:
Conclusion: some made better, some worse. Leaving for now, will delete features later

### 50. Adding different categories based on survival rate of family related: SibSp, Parch, and new feature of Family size (1 + SibSp + 1 Parch)
3 of these features are clearly closely correlated features, we'll remove the less relevant later. (there are 2 more categories below that will also have high correlation, number with same last name, and number with same ticket number).  Can let the models to actually choose what's the most important categories
Line 363: Logistic - liblinear - Stats: Default params cross: 0.838 (+-0.032=0.807), grid train: 0.853, best classifier cross: 0.838 (+-0.032=0.807), time took: 22 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.814 (+-0.03=0.784), grid train: 0.833, best classifier cross: 0.814 (+-0.03=0.784), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.827 (+-0.032=0.795), grid train: 0.871, best classifier cross: 0.827 (+-0.032=0.795), time took: 4 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.838 (+-0.037=0.801), grid train: 0.879, best classifier cross: 0.838 (+-0.037=0.801), time took: 2 sec, best classifier:
	NB                   - Stats: Default params cross: 0.419 (+-0.015=0.404), grid train: 0.433, best classifier cross: 0.419 (+-0.015=0.404), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.841 (+-0.033=0.808), grid train: 0.861, best classifier cross: 0.841 (+-0.035=0.805), time took: 33 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.835 (+-0.036=0.799), grid train: 0.844, best classifier cross: 0.834 (+-0.033=0.8), time took: 32 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.862, best classifier cross: 0.842 (+-0.034=0.807), time took: 13 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.833 (+-0.03=0.803), grid train: 0.946, best classifier cross: 0.833 (+-0.03=0.803), time took: 23 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.886, best classifier cross: 0.838 (+-0.032=0.806), time took: 38 sec, best classifier:
Conclusion: although accuracy is worse, leaving, will remove most correlated and less relevant later

### 51. Splitting Deck a bit better based on how frequence a category is and survival rate
	Logistic - liblinear - Stats: Default params cross: 0.84 (+-0.037=0.803), grid train: 0.856, best classifier cross: 0.84 (+-0.037=0.803), time took: 22 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.812 (+-0.029=0.782), grid train: 0.832, best classifier cross: 0.812 (+-0.029=0.782), time took: 2 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.826 (+-0.034=0.792), grid train: 0.872, best classifier cross: 0.826 (+-0.034=0.792), time took: 4 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.841 (+-0.039=0.801), grid train: 0.878, best classifier cross: 0.841 (+-0.039=0.801), time took: 2 sec, best classifier:
	NB                   - Stats: Default params cross: 0.416 (+-0.016=0.4), grid train: 0.433, best classifier cross: 0.416 (+-0.016=0.4), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.845 (+-0.037=0.809), grid train: 0.864, best classifier cross: 0.844 (+-0.035=0.809), time took: 33 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.837 (+-0.035=0.802), grid train: 0.855, best classifier cross: 0.837 (+-0.037=0.8), time took: 36 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.868, best classifier cross: 0.843 (+-0.036=0.807), time took: 15 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.832 (+-0.034=0.798), grid train: 0.948, best classifier cross: 0.832 (+-0.034=0.798), time took: 24 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.883, best classifier cross: 0.842 (+-0.027=0.815), time took: 38 sec, best classifier:
Conclusion: Does't make much of a difference at this point, but nicer code

### 52. Adding category of Fare per ticket, since it was proven that Fare appears for the whole party. Also binned into categories based on KDE and surival rates
	Logistic - liblinear - Stats: Default params cross: 0.84 (+-0.026=0.813), grid train: 0.86, best classifier cross: 0.84 (+-0.026=0.813), time took: 18 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.809 (+-0.034=0.775), grid train: 0.841, best classifier cross: 0.809 (+-0.034=0.775), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.82 (+-0.035=0.785), grid train: 0.87, best classifier cross: 0.82 (+-0.035=0.785), time took: 3 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.828 (+-0.036=0.792), grid train: 0.88, best classifier cross: 0.828 (+-0.036=0.792), time took: 3 sec, best classifier:
	NB                   - Stats: Default params cross: 0.541 (+-0.061=0.48), grid train: 0.558, best classifier cross: 0.541 (+-0.061=0.48), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.843 (+-0.036=0.807), grid train: 0.868, best classifier cross: 0.843 (+-0.035=0.808), time took: 30 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.838 (+-0.037=0.802), grid train: 0.851, best classifier cross: 0.84 (+-0.035=0.804), time took: 34 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.868, best classifier cross: 0.845 (+-0.038=0.807), time took: 15 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.842 (+-0.029=0.813), grid train: 0.947, best classifier cross: 0.842 (+-0.029=0.813), time took: 29 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.883, best classifier cross: 0.845 (+-0.031=0.814), time took: 36 sec, best classifier:
Conclusion: most didn't change drastically, XGB improved

### 53. Added binning of age - see Advanced feature engineering.ipynb
	Logistic - liblinear - Stats: Default params cross: 0.845 (+-0.028=0.817), grid train: 0.86, best classifier cross: 0.845 (+-0.028=0.817), time took: 21 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.808 (+-0.035=0.774), grid train: 0.833, best classifier cross: 0.808 (+-0.035=0.774), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.826 (+-0.035=0.791), grid train: 0.888, best classifier cross: 0.826 (+-0.035=0.791), time took: 5 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.815 (+-0.035=0.78), grid train: 0.899, best classifier cross: 0.815 (+-0.035=0.78), time took: 4 sec, best classifier:
	NB                   - Stats: Default params cross: 0.613 (+-0.082=0.531), grid train: 0.65, best classifier cross: 0.613 (+-0.082=0.531), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.842 (+-0.035=0.806), grid train: 0.866, best classifier cross: 0.846 (+-0.036=0.81), time took: 43 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.84 (+-0.037=0.802), grid train: 0.856, best classifier cross: 0.836 (+-0.036=0.8), time took: 39 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.866, best classifier cross: 0.844 (+-0.038=0.806), time took: 17 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.833 (+-0.033=0.8), grid train: 0.93, best classifier cross: 0.833 (+-0.033=0.8), time took: 33 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.893, best classifier cross: 0.84 (+-0.028=0.812), time took: 46 sec, best classifier:
Conclusion: not drastically different, obviously some features are not useful, need to remove some

### 54. Add imputing Age based on Regressor based on following features: ['Pclass', 'Title', 'Parch', 'SibSp', 'Ticket_Frequency', 'Embarked'], the rest didn't help to predict
	Logistic - liblinear - Stats: Default params cross: 0.838 (+-0.028=0.81), grid train: 0.859, best classifier cross: 0.838 (+-0.028=0.81), time took: 17 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.806 (+-0.027=0.779), grid train: 0.832, best classifier cross: 0.806 (+-0.027=0.779), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.829 (+-0.037=0.792), grid train: 0.889, best classifier cross: 0.829 (+-0.037=0.792), time took: 4 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.814 (+-0.037=0.777), grid train: 0.899, best classifier cross: 0.814 (+-0.037=0.777), time took: 4 sec, best classifier:
	NB                   - Stats: Default params cross: 0.592 (+-0.074=0.518), grid train: 0.631, best classifier cross: 0.592 (+-0.074=0.518), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.842 (+-0.035=0.806), grid train: 0.868, best classifier cross: 0.844 (+-0.033=0.811), time took: 33 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.838 (+-0.034=0.805), grid train: 0.856, best classifier cross: 0.84 (+-0.037=0.803), time took: 54 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.866, best classifier cross: 0.846 (+-0.035=0.812), time took: 14 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.832 (+-0.037=0.795), grid train: 0.924, best classifier cross: 0.832 (+-0.037=0.795), time took: 31 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.898, best classifier cross: 0.838 (+-0.032=0.807), time took: 40 sec, best classifier:
Conclusion: results very similar, slightly worse.  But the spread is pretty different, less concentrated around means especially of class 3.

### 55. Removing some features with highest correlation
See included file: output\feature_correlations.xlsx 
Consider removing soon:
	- 'Title_Master' - already included in age categories Age_-4, Age_4-11.  However, if we are removing 'Sex', need to double check that survival rate between little girls and boys is not different
	- 'Title_Miss' - same. Consider leaving Sex, and removing 'Title_Master' and 'Title_Miss'
	- 'Sex' - fully included in 'Title' feature
Not removing for now:
	- 'Ticket_Frequency' and 'SibSpBin', 'ParchBin', 'Family size' - high correlation, but not exactly the same thing.  Let's see what features will be important. 
	- 'Family/ticket survival known' and 'SibSpBin', 'ParchBin', 'Family size' - makes sense the higher number of people in the family, the more of a chance we know their survival, but not the same thing
	- 'Title' and 'SibSpBin', 'ParchBin', 'Family size' - makes sense, specific title means more of a chance for specific family
	- 'SibSpBin', 'ParchBin', 'Family size' have connections with each other.  Consider later leaving only 1 or 2 of them.
	- 'SibSpBin', 'ParchBin', 'Family size' and 'FareBin' - perhaps there were sales for tickets for a family.  But correlation is not critically high.
	- 'SibSpBin', 'ParchBin', 'Family size' and 'Age' - makes sense that number of family members connected to Age.  No critical correlations.
	- 'Deck' and 'Pclass' - strong connection.  Need to see which of the two will give better results 
	- 'DeckBin_unknown_T' and 'Fare bin_13.5+' - opposite correlation.  Perhaps need to remove Fare bin_13.5+ since it's included in DeckBin information
	- 'Pclass' and 'Fare bin', especially 'Pclass_1' and 'Fare bin_13.5+', see which one leaving later, classes or Fares
	
### 56. Since most kids and women survived, but some didn't, and most men survived, but some didn't, trying to understand what's different about those men/women
	Logistic - liblinear - Stats: Default params cross: 0.833 (+-0.031=0.802), grid train: 0.859, best classifier cross: 0.833 (+-0.031=0.802), time took: 17 sec, best classifier:
	KNN - 14             - Stats: Default params cross: 0.804 (+-0.022=0.781), grid train: 0.838, best classifier cross: 0.804 (+-0.022=0.781), time took: 1 sec, best classifier:
	SVM - rbf            - Stats: Default params cross: 0.824 (+-0.034=0.79), grid train: 0.888, best classifier cross: 0.824 (+-0.034=0.79), time took: 4 sec, best classifier:
	SVM - poly           - Stats: Default params cross: 0.804 (+-0.039=0.765), grid train: 0.899, best classifier cross: 0.804 (+-0.039=0.765), time took: 4 sec, best classifier:
	NB                   - Stats: Default params cross: 0.634 (+-0.077=0.558), grid train: 0.673, best classifier cross: 0.634 (+-0.077=0.558), time took: 0 sec, best classifier:
	RandomForest - 5     - Stats: Default params cross: 0.838 (+-0.034=0.804), grid train: 0.864, best classifier cross: 0.841 (+-0.03=0.811), time took: 37 sec, best classifier:
	RandomForest - 4     - Stats: Default params cross: 0.84 (+-0.039=0.801), grid train: 0.848, best classifier cross: 0.838 (+-0.037=0.802), time took: 33 sec, best classifier:
	RandomForest Explicit 5 - Stats: Default params cross: grid train: 0.863, best classifier cross: 0.836 (+-0.029=0.807), time took: 14 sec = 0 min, importances:
	XGB                  - Stats: Default params cross: 0.829 (+-0.036=0.793), grid train: 0.929, best classifier cross: 0.829 (+-0.036=0.793), time took: 31 sec, best classifier:
	FINAL                                        - Stats: Best classifiers + Voting train: 0.897, best classifier cross: 0.832 (+-0.032=0.8), time took: 41 sec, best classifier:
Conclusion: Similar, slightly worse results.  Did split up of a few categories of ParchBin, Family size, Deck due to different survival rates

### 57. Before removing features, remember benchmark with random_state 50
	Stats single: {'Name': 'Log', 'Cross accuracy': 0.833, 'STD': 0.031, 'Cross accuracy-STD': 0.802, 'Time sec': 0}
	Stats single: {'Name': 'KNN 14', 'Cross accuracy': 0.804, 'STD': 0.022, 'Cross accuracy-STD': 0.781, 'Time sec': 13}
	Stats single: {'Name': 'SVM rbf', 'Cross accuracy': 0.824, 'STD': 0.034, 'Cross accuracy-STD': 0.79, 'Time sec': 2}
	Stats single: {'Name': 'SVM poly', 'Cross accuracy': 0.803, 'STD': 0.04, 'Cross accuracy-STD': 0.763, 'Time sec': 2}
	Stats single: {'Name': 'NB', 'Cross accuracy': 0.639, 'STD': 0.081, 'Cross accuracy-STD': 0.558, 'Time sec': 0}
	Stats single: {'Name': 'RF 5', 'Cross accuracy': 0.836, 'STD': 0.032, 'Cross accuracy-STD': 0.804, 'Time sec': 14}
	Stats single: {'Name': 'RF 4', 'Cross accuracy': 0.833, 'STD': 0.036, 'Cross accuracy-STD': 0.797, 'Time sec': 13}
	Stats single: {'Name': 'XGB', 'Cross accuracy': 0.833, 'STD': 0.035, 'Cross accuracy-STD': 0.798, 'Time sec': 14}
	Stats single: {'Name': 'Voting soft', 'Cross accuracy': 0.834, 'STD': 0.032, 'Cross accuracy-STD': 0.802, 'Time sec': 49}
	Stats single: {'Name': 'Voting hard', 'Cross accuracy': 0.838, 'STD': 0.036, 'Cross accuracy-STD': 0.803, 'Time sec': 41}
	
### 58. Removing one feature per category from categories where some features are not important
	Stats single: {'Name': 'Log', 'Cross accuracy': 0.835, 'STD': 0.032, 'Cross accuracy-STD': 0.803, 'Time sec': 0}
	Stats single: {'Name': 'KNN 14', 'Cross accuracy': 0.804, 'STD': 0.029, 'Cross accuracy-STD': 0.774, 'Time sec': 13}
	Stats single: {'Name': 'SVM rbf', 'Cross accuracy': 0.827, 'STD': 0.029, 'Cross accuracy-STD': 0.798, 'Time sec': 2}
	Stats single: {'Name': 'SVM poly', 'Cross accuracy': 0.816, 'STD': 0.032, 'Cross accuracy-STD': 0.784, 'Time sec': 2}
	Stats single: {'Name': 'NB', 'Cross accuracy': 0.469, 'STD': 0.04, 'Cross accuracy-STD': 0.429, 'Time sec': 0}
	Stats single: {'Name': 'RF 5', 'Cross accuracy': 0.84, 'STD': 0.035, 'Cross accuracy-STD': 0.805, 'Time sec': 13}
	Stats single: {'Name': 'RF 4', 'Cross accuracy': 0.84, 'STD': 0.034, 'Cross accuracy-STD': 0.805, 'Time sec': 13}
	Stats single: {'Name': 'XGB', 'Cross accuracy': 0.835, 'STD': 0.033, 'Cross accuracy-STD': 0.802, 'Time sec': 10}
	Stats single: {'Name': 'Voting soft', 'Cross accuracy': 0.84, 'STD': 0.036, 'Cross accuracy-STD': 0.803, 'Time sec': 50}
	Stats single: {'Name': 'Voting hard', 'Cross accuracy': 0.837, 'STD': 0.028, 'Cross accuracy-STD': 0.809, 'Time sec': 39}
Conclusion: some models improved slightly
    'minor_columns_to_drop': [
        # -- Embarked - not very important, but at least Embarked_S is place 15-16 in most, consider removing altogether
        'Embarked_Q',  # low in all 4
        # -- Age - not extemely important, most models Age_-4 is important (15), XGB gives more age importance (6,8)
        'Age_4-11',  # low in all 4 (perhaps because of titles that serve same purpose)
        # -- ParchBin - not extremely important in general (besides one exception > 15 in all models). Consider removing altogether
        'ParchBin_3',  # all models very low, in logistic place 9, removing since perhaps logistic overfitting.
        # -- SibSpBin - not extremely important in general (>17 in all models).  Consider removing altogether
        'SibSpBin_4',  # very low in all models
        # -- Family size - seems more important than ParchBin and SibSpBin, but less consistent between models:
        #       - Family size_1 - consistently important (0, 11, 16)
        #       - Family size_2 - consistently not important (>19, and more)
        #       - Family size_3 - consistently not important (>20, and more)
        #       - Family size_4 - consistently not important (>24, and more)
        #       - Family size_567 important in 3 models, not in XGB
        #       - Family size_8+ - extremely low in 3 models, high (6) in logistic
        #       Conclusion: remove 4, later can remove also 3, 2
        'Family size_4',
        # -- Fare bin - mostly not very important, a few important:
        #       - Fare bin_13.5+ - places 2-10
        #       - Fare bin_7.896-7.925 - not consistent, sometimes very important, sometimes not
        #       - For now only removing 'Fare bin_0.1-4' (low in all)
        #       - Consider removing all others
        'Fare bin_0.1-4',
        # -- Deck - some important, some not
        #       - DeckBin_AG - very low in all
        #       - DeckBin_B - low in all
        #       - DeckBin_CF - low in all
        #       - DeckBin_DE - high in all (5-10) - need to leave
        #       - DeckBin_unknown_T - very high in all (3,6,23) - need to leave
        #       Conclusion: for now removing DeckBin_AG, consider removing B and CF also
        'DeckBin_AG',
        # -- 'Family/ticket survival known'  # low in all 4
        'Family/ticket survival known',
        # -- Title - most important in most models: Mr important in all, XGB considers everything besides Mr low. Leaving all
        # -- Pclass - 3 is most important (1,5,8), 1 second (9,19,22 - perhaps have other proxies), 2 - lowest (12,13,24,38).
        #       Conclusion: for now not removing, consider removing 2, and maybe 1 later
        # -- Ticket_Frequency - place 7,10,14, leaving
        # -- Known family/ticket survived % - places 2,4 - one of the most important
        # -- Sex - place 1 in all but XGB (23), leaving]
		
### 59. Removing low importance categories:
	- SibSpBin altogether
	- Age_27-31
	- ParchBin_4+
	- Family size_2
	- Fare bin_0
	- DeckBin_B
	- DeckBin_CF
	Stats single: {'Name': 'Log', 'Cross accuracy': 0.836, 'STD': 0.023, 'Cross accuracy-STD': 0.814, 'Time sec': 0}
	Stats single: {'Name': 'KNN 14', 'Cross accuracy': 0.813, 'STD': 0.028, 'Cross accuracy-STD': 0.785, 'Time sec': 11}
	Stats single: {'Name': 'SVM rbf', 'Cross accuracy': 0.835, 'STD': 0.034, 'Cross accuracy-STD': 0.801, 'Time sec': 1}
	Stats single: {'Name': 'SVM poly', 'Cross accuracy': 0.828, 'STD': 0.038, 'Cross accuracy-STD': 0.79, 'Time sec': 1}
	Stats single: {'Name': 'NB', 'Cross accuracy': 0.593, 'STD': 0.051, 'Cross accuracy-STD': 0.542, 'Time sec': 0}
	Stats single: {'Name': 'RF 5', 'Cross accuracy': 0.844, 'STD': 0.034, 'Cross accuracy-STD': 0.81, 'Time sec': 12}
	Stats single: {'Name': 'RF 4', 'Cross accuracy': 0.843, 'STD': 0.035, 'Cross accuracy-STD': 0.808, 'Time sec': 11}
	Stats single: {'Name': 'XGB', 'Cross accuracy': 0.843, 'STD': 0.034, 'Cross accuracy-STD': 0.809, 'Time sec': 8}
	Stats single: {'Name': 'Voting soft', 'Cross accuracy': 0.837, 'STD': 0.034, 'Cross accuracy-STD': 0.803, 'Time sec': 44}
	Stats single: {'Name': 'Voting hard', 'Cross accuracy': 0.851, 'STD': 0.032, 'Cross accuracy-STD': 0.818, 'Time sec': 71}
Conclusions: improvement!!

### 60. Removing some low categories
Removing the following low importance categories:
- Age_11-24
- ParchBin_2
- Family size_3
- 'Fare bin_12.5-13.5'
- 'Fare bin_4-5'
- 'Fare bin_5-7',
- 'Fare bin_7.925-8.662'
	Stats single: {'Name': 'Log', 'Cross accuracy': 0.836, 'STD': 0.029, 'Cross accuracy-STD': 0.808, 'Time sec': 0}
	Stats single: {'Name': 'KNN 14', 'Cross accuracy': 0.825, 'STD': 0.027, 'Cross accuracy-STD': 0.798, 'Time sec': 15}
	Stats single: {'Name': 'SVM rbf', 'Cross accuracy': 0.841, 'STD': 0.035, 'Cross accuracy-STD': 0.806, 'Time sec': 2}
	Stats single: {'Name': 'SVM poly', 'Cross accuracy': 0.832, 'STD': 0.037, 'Cross accuracy-STD': 0.795, 'Time sec': 1}
	Stats single: {'Name': 'NB', 'Cross accuracy': 0.563, 'STD': 0.046, 'Cross accuracy-STD': 0.517, 'Time sec': 0}
	Stats single: {'Name': 'RF 7', 'Cross accuracy': 0.853, 'STD': 0.028, 'Cross accuracy-STD': 0.825, 'Time sec': 19}
	Stats single: {'Name': 'RF 5', 'Cross accuracy': 0.845, 'STD': 0.034, 'Cross accuracy-STD': 0.811, 'Time sec': 16}
	Stats single: {'Name': 'RF 4', 'Cross accuracy': 0.842, 'STD': 0.038, 'Cross accuracy-STD': 0.804, 'Time sec': 15}
	Stats single: {'Name': 'XGB', 'Cross accuracy': 0.838, 'STD': 0.03, 'Cross accuracy-STD': 0.808, 'Time sec': 12}
	Stats single: {'Name': 'Voting soft', 'Cross accuracy': 0.85, 'STD': 0.036, 'Cross accuracy-STD': 0.814, 'Time sec': 89}
	Stats single: {'Name': 'Voting hard', 'Cross accuracy': 0.849, 'STD': 0.032, 'Cross accuracy-STD': 0.816, 'Time sec': 47}
Conclusions: results similar

### 61. Removing Sex category
Since Sex category has an extremely high correlation with Title, left till now to see if Title will be important.  Since it is, removing Sex.
	Stats single: {'Name': 'Log', 'Cross accuracy': 0.835, 'STD': 0.03, 'Cross accuracy-STD': 0.805, 'Time sec': 0}
	Stats single: {'Name': 'KNN 14', 'Cross accuracy': 0.806, 'STD': 0.032, 'Cross accuracy-STD': 0.774, 'Time sec': 13}
	Stats single: {'Name': 'SVM rbf', 'Cross accuracy': 0.841, 'STD': 0.034, 'Cross accuracy-STD': 0.806, 'Time sec': 1}
	Stats single: {'Name': 'SVM poly', 'Cross accuracy': 0.833, 'STD': 0.034, 'Cross accuracy-STD': 0.798, 'Time sec': 2}
	Stats single: {'Name': 'NB', 'Cross accuracy': 0.497, 'STD': 0.04, 'Cross accuracy-STD': 0.457, 'Time sec': 0}
	Stats single: {'Name': 'RF 7', 'Cross accuracy': 0.854, 'STD': 0.031, 'Cross accuracy-STD': 0.823, 'Time sec': 13}
	Stats single: {'Name': 'RF 5', 'Cross accuracy': 0.841, 'STD': 0.027, 'Cross accuracy-STD': 0.813, 'Time sec': 13}
	Stats single: {'Name': 'RF 4', 'Cross accuracy': 0.846, 'STD': 0.028, 'Cross accuracy-STD': 0.818, 'Time sec': 13}
	Stats single: {'Name': 'XGB', 'Cross accuracy': 0.838, 'STD': 0.039, 'Cross accuracy-STD': 0.799, 'Time sec': 10}
	Stats single: {'Name': 'Voting soft', 'Cross accuracy': 0.844, 'STD': 0.034, 'Cross accuracy-STD': 0.81, 'Time sec': 68}
	Stats single: {'Name': 'Voting hard', 'Cross accuracy': 0.845, 'STD': 0.033, 'Cross accuracy-STD': 0.813, 'Time sec': 44}
Conclusions: results similar, slightly worse. Leaving as is, makes sense and makes model more stable


### 62. Readding all that was removed before Sex removed to make sure that removing Sex didn't change some conclusions.  Use as benchmark later.
	Stats single: {'Name': 'Log', 'Cross accuracy': 0.836, 'STD': 0.031, 'Cross accuracy-STD': 0.805, 'Time sec': 0}
	Stats single: {'Name': 'KNN 14', 'Cross accuracy': 0.775, 'STD': 0.032, 'Cross accuracy-STD': 0.742, 'Time sec': 12}
	Stats single: {'Name': 'SVM rbf', 'Cross accuracy': 0.824, 'STD': 0.032, 'Cross accuracy-STD': 0.792, 'Time sec': 2}
	Stats single: {'Name': 'SVM poly', 'Cross accuracy': 0.792, 'STD': 0.028, 'Cross accuracy-STD': 0.765, 'Time sec': 2}
	Stats single: {'Name': 'NB', 'Cross accuracy': 0.605, 'STD': 0.072, 'Cross accuracy-STD': 0.533, 'Time sec': 0}
	Stats single: {'Name': 'RF 7', 'Cross accuracy': 0.842, 'STD': 0.031, 'Cross accuracy-STD': 0.811, 'Time sec': 14}
	Stats single: {'Name': 'RF 5', 'Cross accuracy': 0.842, 'STD': 0.029, 'Cross accuracy-STD': 0.813, 'Time sec': 13}
	Stats single: {'Name': 'RF 4', 'Cross accuracy': 0.841, 'STD': 0.034, 'Cross accuracy-STD': 0.807, 'Time sec': 13}
	Stats single: {'Name': 'XGB', 'Cross accuracy': 0.833, 'STD': 0.035, 'Cross accuracy-STD': 0.797, 'Time sec': 12}
	Stats single: {'Name': 'Voting soft', 'Cross accuracy': 0.835, 'STD': 0.033, 'Cross accuracy-STD': 0.802, 'Time sec': 85}
	Stats single: {'Name': 'Voting hard', 'Cross accuracy': 0.84, 'STD': 0.033, 'Cross accuracy-STD': 0.806, 'Time sec': 62}
Conclusions: continue removing what was already removed: 'Family/ticket survival known', 'SibSp' 
	Stats single: {'Name': 'Log', 'Cross accuracy': 0.829, 'STD': 0.029, 'Cross accuracy-STD': 0.801, 'Time sec': 0}
	Stats single: {'Name': 'KNN 14', 'Cross accuracy': 0.783, 'STD': 0.023, 'Cross accuracy-STD': 0.761, 'Time sec': 15}
	Stats single: {'Name': 'SVM rbf', 'Cross accuracy': 0.826, 'STD': 0.032, 'Cross accuracy-STD': 0.794, 'Time sec': 2}
	Stats single: {'Name': 'SVM poly', 'Cross accuracy': 0.806, 'STD': 0.021, 'Cross accuracy-STD': 0.785, 'Time sec': 2}
	Stats single: {'Name': 'NB', 'Cross accuracy': 0.734, 'STD': 0.085, 'Cross accuracy-STD': 0.65, 'Time sec': 0}
	Stats single: {'Name': 'RF 7', 'Cross accuracy': 0.844, 'STD': 0.034, 'Cross accuracy-STD': 0.81, 'Time sec': 15}
	Stats single: {'Name': 'RF 5', 'Cross accuracy': 0.842, 'STD': 0.037, 'Cross accuracy-STD': 0.804, 'Time sec': 16}
	Stats single: {'Name': 'RF 4', 'Cross accuracy': 0.838, 'STD': 0.03, 'Cross accuracy-STD': 0.809, 'Time sec': 12}
	Stats single: {'Name': 'XGB', 'Cross accuracy': 0.832, 'STD': 0.031, 'Cross accuracy-STD': 0.8, 'Time sec': 11}
	Stats single: {'Name': 'Voting soft', 'Cross accuracy': 0.837, 'STD': 0.03, 'Cross accuracy-STD': 0.808, 'Time sec': 76}
	Stats single: {'Name': 'Voting hard', 'Cross accuracy': 0.85, 'STD': 0.034, 'Cross accuracy-STD': 0.816, 'Time sec': 48}
Conclusions: mostly better

### 63: Removing Embarked_Q (low importance), Parch (low importance, and high correlation with family size)
	Stats single: {'Name': 'Log', 'Cross accuracy': 0.838, 'STD': 0.025, 'Cross accuracy-STD': 0.813, 'Time sec': 0}
	Stats single: {'Name': 'KNN 14', 'Cross accuracy': 0.773, 'STD': 0.031, 'Cross accuracy-STD': 0.743, 'Time sec': 14}
	Stats single: {'Name': 'SVM rbf', 'Cross accuracy': 0.836, 'STD': 0.036, 'Cross accuracy-STD': 0.8, 'Time sec': 2}
	Stats single: {'Name': 'SVM poly', 'Cross accuracy': 0.827, 'STD': 0.023, 'Cross accuracy-STD': 0.804, 'Time sec': 2}
	Stats single: {'Name': 'NB', 'Cross accuracy': 0.747, 'STD': 0.08, 'Cross accuracy-STD': 0.667, 'Time sec': 0}
	Stats single: {'Name': 'RF 7', 'Cross accuracy': 0.851, 'STD': 0.028, 'Cross accuracy-STD': 0.823, 'Time sec': 14}
	Stats single: {'Name': 'RF 5', 'Cross accuracy': 0.843, 'STD': 0.033, 'Cross accuracy-STD': 0.81, 'Time sec': 14}
	Stats single: {'Name': 'RF 4', 'Cross accuracy': 0.841, 'STD': 0.032, 'Cross accuracy-STD': 0.808, 'Time sec': 13}
	Stats single: {'Name': 'XGB', 'Cross accuracy': 0.842, 'STD': 0.031, 'Cross accuracy-STD': 0.811, 'Time sec': 13}
	Stats single: {'Name': 'Voting soft', 'Cross accuracy': 0.842, 'STD': 0.032, 'Cross accuracy-STD': 0.81, 'Time sec': 73}
	Stats single: {'Name': 'Voting hard', 'Cross accuracy': 0.853, 'STD': 0.033, 'Cross accuracy-STD': 0.82, 'Time sec': 45}
Conclusions: much better.  Removing many more fields: 
	'Age_4-11','Age_27-31','Age_11-24','Age_31-32','Age_48-57'
	'Family size_4'
	'Fare bin_0.1-4',
	'Fare bin_0','Fare bin_12.5-13.5','Fare bin_4-5','Fare bin_5-7','Fare bin_7.925-8.662','Fare bin_7.796-7.896',
	'DeckBin_AG'
	Stats single: {'Name': 'Log', 'Cross accuracy': 0.84, 'STD': 0.026, 'Cross accuracy-STD': 0.814, 'Time sec': 0}
	Stats single: {'Name': 'KNN 14', 'Cross accuracy': 0.816, 'STD': 0.035, 'Cross accuracy-STD': 0.781, 'Time sec': 14}
	Stats single: {'Name': 'SVM rbf', 'Cross accuracy': 0.846, 'STD': 0.037, 'Cross accuracy-STD': 0.809, 'Time sec': 2}
	Stats single: {'Name': 'SVM poly', 'Cross accuracy': 0.826, 'STD': 0.037, 'Cross accuracy-STD': 0.789, 'Time sec': 1}
	Stats single: {'Name': 'NB', 'Cross accuracy': 0.586, 'STD': 0.041, 'Cross accuracy-STD': 0.544, 'Time sec': 0}
	Stats single: {'Name': 'RF 7', 'Cross accuracy': 0.854, 'STD': 0.029, 'Cross accuracy-STD': 0.825, 'Time sec': 13}
	Stats single: {'Name': 'RF 5', 'Cross accuracy': 0.847, 'STD': 0.03, 'Cross accuracy-STD': 0.818, 'Time sec': 14}
	Stats single: {'Name': 'RF 4', 'Cross accuracy': 0.842, 'STD': 0.029, 'Cross accuracy-STD': 0.812, 'Time sec': 16}
	Stats single: {'Name': 'XGB', 'Cross accuracy': 0.848, 'STD': 0.027, 'Cross accuracy-STD': 0.822, 'Time sec': 10}
	Stats single: {'Name': 'Voting soft', 'Cross accuracy': 0.846, 'STD': 0.036, 'Cross accuracy-STD': 0.811, 'Time sec': 66}
	Stats single: {'Name': 'Voting hard', 'Cross accuracy': 0.852, 'STD': 0.029, 'Cross accuracy-STD': 0.823, 'Time sec': 47}
Conclusions: much better

### 64. Removing many other features
Regarding following features with high correlation: 
	- Title_Master and Age_-4 - although there is a high correlation, both important and don't overlap because of girls, and masters being > 4 also
	- Fare bin_13.5+ and Pclass_1 - although there is a high correlation they don't completely overlap and each predicts survival
	- Decks 'B','C','D','E','F' and high correlation to 1st class and negative correlation to class 3 - it's not the same thing, since some decks were mixed (DEF), 
		and even same class might have different survival per Deck.  So leaving both.
Removing the following low importance features:
	- Embarked completely - low importance mostly, and probably connection comes due to high correlation between embarkation and class
	- Age_40-48, Age_57+
	- Family size_2, Family size_3
	- DeckBin_B, DeckBin_CF. Btw, seems that the reasons Decks D, E are important is due to different survival rate there because it's a mixed cabin
	Stats single: {'Name': 'Log', 'Cross accuracy': 0.84, 'STD': 0.032, 'Cross accuracy-STD': 0.807, 'Time sec': 0}
	Stats single: {'Name': 'KNN 14', 'Cross accuracy': 0.825, 'STD': 0.03, 'Cross accuracy-STD': 0.794, 'Time sec': 11}
	Stats single: {'Name': 'SVM rbf', 'Cross accuracy': 0.848, 'STD': 0.035, 'Cross accuracy-STD': 0.814, 'Time sec': 1}
	Stats single: {'Name': 'SVM poly', 'Cross accuracy': 0.846, 'STD': 0.029, 'Cross accuracy-STD': 0.817, 'Time sec': 1}
	Stats single: {'Name': 'NB', 'Cross accuracy': 0.435, 'STD': 0.021, 'Cross accuracy-STD': 0.415, 'Time sec': 0}
	Stats single: {'Name': 'RF 7', 'Cross accuracy': 0.858, 'STD': 0.029, 'Cross accuracy-STD': 0.828, 'Time sec': 13}
	Stats single: {'Name': 'RF 5', 'Cross accuracy': 0.843, 'STD': 0.027, 'Cross accuracy-STD': 0.816, 'Time sec': 14}
	Stats single: {'Name': 'RF 4', 'Cross accuracy': 0.843, 'STD': 0.027, 'Cross accuracy-STD': 0.816, 'Time sec': 14}
	Stats single: {'Name': 'XGB', 'Cross accuracy': 0.849, 'STD': 0.028, 'Cross accuracy-STD': 0.821, 'Time sec': 7}
	Stats single: {'Name': 'Voting soft', 'Cross accuracy': 0.846, 'STD': 0.032, 'Cross accuracy-STD': 0.815, 'Time sec': 63}
	Stats single: {'Name': 'Voting hard', 'Cross accuracy': 0.847, 'STD': 0.029, 'Cross accuracy-STD': 0.819, 'Time sec': 51}
Conclusions: Similar, but prevents overfitting so presumably will give better answers on the website

### 65. Removing NB classifer since it consistently gives worse results
	Stats single: {'Name': 'Log', 'Cross accuracy': 0.84, 'STD': 0.032, 'Cross accuracy-STD': 0.807, 'Time sec': 0}
	Stats single: {'Name': 'KNN 14', 'Cross accuracy': 0.825, 'STD': 0.03, 'Cross accuracy-STD': 0.794, 'Time sec': 12}
	Stats single: {'Name': 'SVM rbf', 'Cross accuracy': 0.847, 'STD': 0.034, 'Cross accuracy-STD': 0.813, 'Time sec': 1}
	Stats single: {'Name': 'SVM poly', 'Cross accuracy': 0.843, 'STD': 0.027, 'Cross accuracy-STD': 0.815, 'Time sec': 1}
	Stats single: {'Name': 'RF 7', 'Cross accuracy': 0.856, 'STD': 0.028, 'Cross accuracy-STD': 0.828, 'Time sec': 14}
	Stats single: {'Name': 'RF 5', 'Cross accuracy': 0.843, 'STD': 0.027, 'Cross accuracy-STD': 0.816, 'Time sec': 12}
	Stats single: {'Name': 'RF 4', 'Cross accuracy': 0.842, 'STD': 0.029, 'Cross accuracy-STD': 0.813, 'Time sec': 12}
	Stats single: {'Name': 'XGB', 'Cross accuracy': 0.846, 'STD': 0.026, 'Cross accuracy-STD': 0.82, 'Time sec': 7}
	Stats single: {'Name': 'Voting soft', 'Cross accuracy': 0.852, 'STD': 0.031, 'Cross accuracy-STD': 0.821, 'Time sec': 52}
	Stats single: {'Name': 'Voting hard', 'Cross accuracy': 0.847, 'STD': 0.031, 'Cross accuracy-STD': 0.817, 'Time sec': 46}
Conclusions: Slightly improved soft and made slightly worse hard (perhaps was helping in special cases, like not to overfit).  Perhaps reconsider adding later.

### 66. Check what else can remove
Conclusion: not removing anything.
Regarding highly correlated, checked and besides comments in commit #64 above, nothing else to do
Also, no categories that are clearly not important in all classifiers.
Some interesting coefficients (taken from Logistic Regression):
	- Age: 	Little kids had a higher chance, 24-40 had a higher chance, with 26-27 being the highest
			Age_26-27    0.194947
			Age_-4    0.131562
			Age_32-40    0.032291
			Age_24-26    0.016200
	- Family size:  Person alone had the highest chance, large families had lower chance of survival, the larger the family the smaller chance
			Family size_1    0.434602
            Family size_567   -0.453681
            Family size_8+   -0.599514
	- Fares:  Highest fare had the highest chance to survive, there is something special about categories of Fare bin_7.896-7.925, Fare bin_7-7.796.  
			Fare bin_13.5+    0.466894
            Fare bin_7.896-7.925    0.323510
            Fare bin_7-7.796    0.233029
            Fare bin_8.662-12.5   -0.019845			# Has much more effect in XGB
		Warning: it could be overfitting, since we don't know to explain why these categories have this different change from categories next to it.
		TODO: consider removing 3 above besides 13.5+.  Although it could be that they explain something else that we removed: specific Deck etc., so need to be careful.
	- Deck: DE - higher chance, perhaps because those were mixed cabins, class 3 had a higher chance than in other places.
		DeckBin_DE    0.252210
        DeckBin_unknown_T   -0.186332
	- Title:  Known - men had lower chance, women and children higher.  Perhaps the reason Miss and Master don't have such high numbers is that other features
			took some importance from them, like age etc.
		Title_Mrs    0.702535
        Title_Miss    0.474255
        Title_Master    0.397960
        Title_Mr   -1.065580
	- Pclass: Known: 1 - highest chance, 2 - high, 3 - low.  Perhaps 1 appears low here since other features took some of the significance: Decks, Fare bins etc.
		Pclass_2    0.202671
        Pclass_1    0.057393
        Pclass_3   -0.214395
	- Ticket_Frequency    0.209756 - this is where Logistic Regression is not accurate, since the relationship with ticket frequency is not linear, see family size above
	- Known family/ticket survived %    0.711410 - extremely important, families mostly either survived together or died together
	
### 67. Adding Grid with some basic configurations.
Results:
	Name	Cross accuracy	STD		Cross accuracy-STD	Time sec
	Log		0.84			0.032		0.807			0
	KNN 14	0.825			0.03		0.794			9
	SVM rbf	0.847			0.034		0.813			1
	SVM poly	0.843		0.027		0.815			1
	RF 7	0.856			0.028		0.828			12
	RF 5	0.843			0.027		0.816			12
	RF 4	0.842			0.029		0.813			12
	XGB		0.846			0.026		0.82			4
	Voting soft single	0.852	0.031	0.821			46
	Voting hard single	0.847	0.031	0.817			41
	Grid Log	0.84		0.032		0.807			6
	Grid KNN	0.843		0.024		0.819			4
	Grid SVM	0.853		0.028		0.825			26
	Grid RF		0.86		0.028		0.832			158
	Grid XGB	0.851		0.026		0.825			122
	Voting soft with grid	0.852	0.034	0.818		650
	Voting hard with grid	0.853	0.025	0.828		46

Best results: Grid RF, RF 7, Voting hard with grid
Worst results: KNN 14, Log and Grid Log, RF 4, RF 5
Correlations (see classifiers_correlations file):
	- Log is not highly correlated with anything (but 1.00 correlation with and without grid, leave without grid), so even though doesn't give great results, perhaps better to leave because of low correlation
	- KNN Grid - not highly correlated with anything. KNN 14 gives bad results (KNN grid chooses 5 and not 14 neighbors). KNN Grid gives decent results.
	- SVM poly - doesn't give great results, but low correlation with others
	- RF 5, RF 4, RF 7 - low correlation with everything.  
	- XGB - low correlation with everything, even not extremely high between XGB default and Grid
	- Voting soft single - high correlation with RFs and SVM rbf.
	- Voting hard single - goes mostly with the RF
	- Voting soft everything - takes mostly SVM rbf
	- Voting hard everything - RF and SVM
Remove:
	- Grid Log - to return if adding more hyperparameters
	- KNN 14 - KNN Grid gives much better results
	- SVM rbg - remove, leave Grid
	- RF 4, RF 5 - not great results, very high correlation with RF 7 and Grid RF
	- Voting soft single, Voting hard single. Results are not better than with the grid, especially for hard.  Just do 1 voting with everything
Keep: 
	- SVM poly - doesn't give great results, but low correlation with others
	- RF 7 - since Grid RF chooses RF 8, and it might be overfitting, leave RF 7 separately
	- XGB - even though Grid XGB does better, leave it since they are not extremely correlated, and Grid might be overfitting
Output best: RF 7, Grid SVM, Grid RF, Grid XGB, Voting soft with grid, Voting hard with grid

### 68 Submit conclusions of all the work online, add important metrics locally
	Also look at results.csv
	Soft & hard - 78.9 - seems lower than RFs because take all the overfitting algorithms
	RF Grid - 80.3 - slightly higher, perhaps because taking higher number of maximum depth (9?)
	RF 7 - 79.9 - highest result of a single classifier
	XGB Grid - 74.6 - seems overfits a lot? Perhaps choosing specific bins of age and fare was good for Random Forest, but not for XGB/SVM 
	SVM Grid - 76.1 - seems also overfits? Perhaps choosing specific bins of age and fare was good for Random Forest, but not for XGB/SVM

	Stats Single: {'Name': 'Log', 'Single accuracy': 0.848, 'Cross accuracy': 0.832, 'STD': 0.027, 'Cross accuracy-STD*2': 0.777, 'Cross accuracy-STD*3': 0.75, 'Overfitting danger': 0.014, 'Time sec': 0}
	Stats Single: {'Name': 'KNN 14', 'Single accuracy': 0.837, 'Cross accuracy': 0.821, 'STD': 0.03, 'Cross accuracy-STD*2': 0.76, 'Cross accuracy-STD*3': 0.729, 'Overfitting danger': 0.014, 'Time sec': 12}
	Stats Single: {'Name': 'SVM rbf', 'Single accuracy': 0.884, 'Cross accuracy': 0.842, 'STD': 0.02, 'Cross accuracy-STD*2': 0.801, 'Cross accuracy-STD*3': 0.781, 'Overfitting danger': 0.038, 'Time sec': 0}
	Stats Single: {'Name': 'SVM poly', 'Single accuracy': 0.886, 'Cross accuracy': 0.849, 'STD': 0.021, 'Cross accuracy-STD*2': 0.806, 'Cross accuracy-STD*3': 0.784, 'Overfitting danger': 0.033, 'Time sec': 0}
	Stats Single: {'Name': 'NB', 'Single accuracy': 0.438, 'Cross accuracy': 0.433, 'STD': 0.021, 'Cross accuracy-STD*2': 0.39, 'Cross accuracy-STD*3': 0.369, 'Overfitting danger': 0.002, 'Time sec': 0}
	Stats Single: {'Name': 'RF 8', 'Single accuracy': 0.895, 'Cross accuracy': 0.853, 'STD': 0.023, 'Cross accuracy-STD*2': 0.806, 'Cross accuracy-STD*3': 0.783, 'Overfitting danger': 0.037, 'Time sec': 9}
	Stats Single: {'Name': 'RF 7', 'Single accuracy': 0.887, 'Cross accuracy': 0.856, 'STD': 0.019, 'Cross accuracy-STD*2': 0.819, 'Cross accuracy-STD*3': 0.8, 'Overfitting danger': 0.027, 'Time sec': 8}
	Stats Single: {'Name': 'RF 6', 'Single accuracy': 0.878, 'Cross accuracy': 0.846, 'STD': 0.019, 'Cross accuracy-STD*2': 0.809, 'Cross accuracy-STD*3': 0.791, 'Overfitting danger': 0.028, 'Time sec': 7}
	Stats Single: {'Name': 'XGB', 'Single accuracy': 0.907, 'Cross accuracy': 0.849, 'STD': 0.017, 'Cross accuracy-STD*2': 0.815, 'Cross accuracy-STD*3': 0.799, 'Overfitting danger': 0.053, 'Time sec': 4}
	Stats Grid: {'Name': 'Grid KNN', 'Single accuracy': 0.837, 'Cross accuracy': 0.829, 'STD': 0.023, 'Cross accuracy-STD*2': 0.784, 'Cross accuracy-STD*3': 0.762, 'Overfitting danger': 0.007, 'Time sec': 12}
	Stats Grid: {'Name': 'Grid SVM', 'Single accuracy': 0.891, 'Cross accuracy': 0.853, 'STD': 0.022, 'Cross accuracy-STD*2': 0.81, 'Cross accuracy-STD*3': 0.788, 'Overfitting danger': 0.034, 'Time sec': 13}
	Stats Grid: {'Name': 'Grid RF', 'Single accuracy': 0.887, 'Cross accuracy': 0.856, 'STD': 0.019, 'Cross accuracy-STD*2': 0.819, 'Cross accuracy-STD*3': 0.8, 'Overfitting danger': 0.027, 'Time sec': 33}
	Stats Grid: {'Name': 'Grid XGB', 'Single accuracy': 0.886, 'Cross accuracy': 0.85, 'STD': 0.021, 'Cross accuracy-STD*2': 0.807, 'Cross accuracy-STD*3': 0.786, 'Overfitting danger': 0.032, 'Time sec': 15}
	Stats Voting: {'Name': 'Voting soft with grid', 'Single accuracy': 0.891, 'Cross accuracy': 0.852, 'STD': 0.019, 'Cross accuracy-STD*2': 0.813, 'Cross accuracy-STD*3': 0.794, 'Overfitting danger': 0.035, 'Time sec': 33}
	Stats Voting: {'Name': 'Voting hard with grid', 'Single accuracy': 0.889, 'Cross accuracy': 0.854, 'STD': 0.021, 'Cross accuracy-STD*2': 0.812, 'Cross accuracy-STD*3': 0.79, 'Overfitting danger': 0.031, 'Time sec': 34}


	Made some results adjustments that will help doing forward:
	- To avoid overfitting, changed scross validation from cv=10 to cv=5, will punish more overfitting algorithms
	- Added overfitting danger estimator = (score of full training - score of cross validation) \* score of full training. 
		Explanation: the bigger the difference of full score that voting algorithms later rely on to cross validation score, the more of a danger of overfitting
		The bigger the score is, the higher danger of soft voting taking the results of this algorithm specifically.
	- Returned different classifiers that were previously removed: KNN 14, SVM rbf without Grid, SVM poly without Grid, NB
	- Added different Random Forest depths to see how they behave with cross validation and new metrics: 8 and 6 (since 7 is currently the best)

	Next steps: Make sure soft voting doesn't get those with high danger of overfitting, make sure that hard voting doesn't get those with high danger of underfitting

 ### 69 Remove from taking into consideration for soft voting classifiers with high chance of overfitting (currently only XGB) and from taking into consideration for hard voting classifier with very low scores (currently only NB)
	Name			Single accuracy		Cross accuracy		STD		Cross accuracy-STD*2	Cross accuracy-STD*3	Overfitting danger	Time sec
Voting soft with grid	0.891				0.85			0.019			0.812					0.794				0.037			28
Voting hard with grid	0.892				0.855			0.022			0.811					0.789				0.033			30
Conclusions: Didn't seem to help, perhaps removing only 1 doesn't help since doesn't have that big of an effect (was voted down anyways).

### 70. Removing more modes from consideration of Soft and Hard Voting (with danger of overfitting for 1st, and worse performing models for 2nd).
	Name				Single accuracy		Cross accuracy		STD		Cross accuracy-STD*2	Cross accuracy-STD*3	Overfitting danger	Time sec
	Voting soft with grid	0.888				0.852			0.023			0.805					0.782				0.032			23
	Voting hard with grid	0.893				0.856			0.019			0.819					0.8					0.033			29
Conclusion: seems to help with hard voting, but not with soft voting.  Reverting soft changes, leaving hard.
	Name				Single accuracy		Cross accuracy		STD		Cross accuracy-STD*2	Cross accuracy-STD*3	Overfitting danger	Time sec
	Voting soft with grid	0.892				0.849			0.024			0.8						0.776				0.039			25
	Voting hard with grid	0.895				0.859			0.018			0.822					0.804				0.032			25
Conclusion: getting great results with hard voting, worse with soft.  Leaving like this for now.

### 71. Adding layer of ensembling on top of separate classifiers, grid, and voting of finding the best answers.
	Stats Single: {'Name': 'Log', 'Single accuracy': 0.848, 'Cross accuracy': 0.832, 'STD': 0.027, 'Cross accuracy-STD*2': 0.777, 'Cross accuracy-STD*3': 0.75, 'Overfitting danger': 0.014, 'Time sec': 0}
	Stats Single: {'Name': 'KNN 14', 'Single accuracy': 0.837, 'Cross accuracy': 0.819, 'STD': 0.031, 'Cross accuracy-STD*2': 0.757, 'Cross accuracy-STD*3': 0.726, 'Overfitting danger': 0.015, 'Time sec': 8}
	Stats Single: {'Name': 'SVM rbf', 'Single accuracy': 0.884, 'Cross accuracy': 0.842, 'STD': 0.02, 'Cross accuracy-STD*2': 0.801, 'Cross accuracy-STD*3': 0.781, 'Overfitting danger': 0.038, 'Time sec': 0}
	Stats Single: {'Name': 'SVM poly', 'Single accuracy': 0.886, 'Cross accuracy': 0.849, 'STD': 0.021, 'Cross accuracy-STD*2': 0.806, 'Cross accuracy-STD*3': 0.784, 'Overfitting danger': 0.033, 'Time sec': 0}
	Stats Single: {'Name': 'NB', 'Single accuracy': 0.438, 'Cross accuracy': 0.433, 'STD': 0.021, 'Cross accuracy-STD*2': 0.39, 'Cross accuracy-STD*3': 0.369, 'Overfitting danger': 0.002, 'Time sec': 0}
	Stats Single: {'Name': 'RF 8', 'Single accuracy': 0.895, 'Cross accuracy': 0.853, 'STD': 0.023, 'Cross accuracy-STD*2': 0.806, 'Cross accuracy-STD*3': 0.783, 'Overfitting danger': 0.037, 'Time sec': 12}
	Stats Single: {'Name': 'RF 7', 'Single accuracy': 0.887, 'Cross accuracy': 0.856, 'STD': 0.019, 'Cross accuracy-STD*2': 0.819, 'Cross accuracy-STD*3': 0.8, 'Overfitting danger': 0.027, 'Time sec': 15}
	Stats Single: {'Name': 'RF 6', 'Single accuracy': 0.878, 'Cross accuracy': 0.846, 'STD': 0.019, 'Cross accuracy-STD*2': 0.809, 'Cross accuracy-STD*3': 0.791, 'Overfitting danger': 0.028, 'Time sec': 200}
	Stats Single: {'Name': 'XGB', 'Single accuracy': 0.903, 'Cross accuracy': 0.844, 'STD': 0.02, 'Cross accuracy-STD*2': 0.804, 'Cross accuracy-STD*3': 0.784, 'Overfitting danger': 0.054, 'Time sec': 4}
	Stats Grid: {'Name': 'Grid KNN', 'Single accuracy': 0.833, 'Cross accuracy': 0.831, 'STD': 0.018, 'Cross accuracy-STD*2': 0.794, 'Cross accuracy-STD*3': 0.776, 'Overfitting danger': 0.002, 'Time sec': 11}
	Stats Grid: {'Name': 'Grid SVM', 'Single accuracy': 0.891, 'Cross accuracy': 0.853, 'STD': 0.022, 'Cross accuracy-STD*2': 0.81, 'Cross accuracy-STD*3': 0.788, 'Overfitting danger': 0.034, 'Time sec': 11}
	Stats Grid: {'Name': 'Grid RF', 'Single accuracy': 0.887, 'Cross accuracy': 0.856, 'STD': 0.019, 'Cross accuracy-STD*2': 0.819, 'Cross accuracy-STD*3': 0.8, 'Overfitting danger': 0.027, 'Time sec': 30}
	Stats Grid: {'Name': 'Grid XGB', 'Single accuracy': 0.886, 'Cross accuracy': 0.847, 'STD': 0.022, 'Cross accuracy-STD*2': 0.803, 'Cross accuracy-STD*3': 0.781, 'Overfitting danger': 0.034, 'Time sec': 15}
	Stats Voting: {'Name': 'Voting soft with grid', 'Single accuracy': 0.891, 'Cross accuracy': 0.849, 'STD': 0.021, 'Cross accuracy-STD*2': 0.807, 'Cross accuracy-STD*3': 0.787, 'Overfitting danger': 0.038, 'Time sec': 41}
	Stats Voting: {'Name': 'Voting hard with grid', 'Single accuracy': 0.892, 'Cross accuracy': 0.858, 'STD': 0.019, 'Cross accuracy-STD*2': 0.82, 'Cross accuracy-STD*3': 0.802, 'Overfitting danger': 0.031, 'Time sec': 33}
	Stats Emsemble: {'Name': 'Ensemble RF', 'Single accuracy': 0.917, 'Cross accuracy': 0.882, 'STD': 0.022, 'Cross accuracy-STD*2': 0.837, 'Cross accuracy-STD*3': 0.815, 'Overfitting danger': 0.032, 'Time sec': 11}
Conclusion: Checking the answers online, despite very high accuracy locally, on the website actually got a pretty bad accuray (.75), 
	can also see it from high correlation with SVM, Soft voting and XGB which we know don't give such great results currently
	
### 72. Back to feature engineering - remove all non-important features for Random Forest
Fare seems not be needed by Random Forest which is currently the best classifier, besides fare 13+.
Also 'Family size_8+' seems low importance (perhaps due to very low number of such people), currently removing. Consider combining with 5,6,7
	Stats Single: {'Name': 'Log', 'Single accuracy': 0.844, 'Cross accuracy': 0.831, 'STD': 0.022, 'Cross accuracy-STD*2': 0.787, 'Cross accuracy-STD*3': 0.765, 'Overfitting danger': 0.011, 'Time sec': 0}
	Stats Single: {'Name': 'KNN 14', 'Single accuracy': 0.838, 'Cross accuracy': 0.817, 'STD': 0.026, 'Cross accuracy-STD*2': 0.765, 'Cross accuracy-STD*3': 0.738, 'Overfitting danger': 0.018, 'Time sec': 12}
	Stats Single: {'Name': 'SVM rbf', 'Single accuracy': 0.868, 'Cross accuracy': 0.84, 'STD': 0.017, 'Cross accuracy-STD*2': 0.806, 'Cross accuracy-STD*3': 0.79, 'Overfitting danger': 0.024, 'Time sec': 1}
	Stats Single: {'Name': 'SVM poly', 'Single accuracy': 0.874, 'Cross accuracy': 0.827, 'STD': 0.022, 'Cross accuracy-STD*2': 0.782, 'Cross accuracy-STD*3': 0.76, 'Overfitting danger': 0.041, 'Time sec': 1}
	Stats Single: {'Name': 'NB', 'Single accuracy': 0.79, 'Cross accuracy': 0.786, 'STD': 0.051, 'Cross accuracy-STD*2': 0.685, 'Cross accuracy-STD*3': 0.634, 'Overfitting danger': 0.003, 'Time sec': 0}
	Stats Single: {'Name': 'RF 8', 'Single accuracy': 0.89, 'Cross accuracy': 0.85, 'STD': 0.018, 'Cross accuracy-STD*2': 0.814, 'Cross accuracy-STD*3': 0.797, 'Overfitting danger': 0.036, 'Time sec': 14}
	Stats Single: {'Name': 'RF 7', 'Single accuracy': 0.888, 'Cross accuracy': 0.851, 'STD': 0.015, 'Cross accuracy-STD*2': 0.82, 'Cross accuracy-STD*3': 0.805, 'Overfitting danger': 0.033, 'Time sec': 14}
	Stats Single: {'Name': 'RF 6', 'Single accuracy': 0.877, 'Cross accuracy': 0.851, 'STD': 0.017, 'Cross accuracy-STD*2': 0.816, 'Cross accuracy-STD*3': 0.799, 'Overfitting danger': 0.023, 'Time sec': 12}
	Stats Single: {'Name': 'XGB', 'Single accuracy': 0.893, 'Cross accuracy': 0.845, 'STD': 0.016, 'Cross accuracy-STD*2': 0.814, 'Cross accuracy-STD*3': 0.798, 'Overfitting danger': 0.043, 'Time sec': 4}
	Stats Grid: {'Name': 'Grid KNN', 'Single accuracy': 0.842, 'Cross accuracy': 0.829, 'STD': 0.016, 'Cross accuracy-STD*2': 0.797, 'Cross accuracy-STD*3': 0.78, 'Overfitting danger': 0.01, 'Time sec': 13}
	Stats Grid: {'Name': 'Grid SVM', 'Single accuracy': 0.883, 'Cross accuracy': 0.845, 'STD': 0.017, 'Cross accuracy-STD*2': 0.811, 'Cross accuracy-STD*3': 0.794, 'Overfitting danger': 0.034, 'Time sec': 13}
	Stats Grid: {'Name': 'Grid RF', 'Single accuracy': 0.877, 'Cross accuracy': 0.851, 'STD': 0.017, 'Cross accuracy-STD*2': 0.816, 'Cross accuracy-STD*3': 0.799, 'Overfitting danger': 0.023, 'Time sec': 27}
	Stats Grid: {'Name': 'Grid XGB', 'Single accuracy': 0.881, 'Cross accuracy': 0.85, 'STD': 0.018, 'Cross accuracy-STD*2': 0.813, 'Cross accuracy-STD*3': 0.794, 'Overfitting danger': 0.028, 'Time sec': 10}
	Stats Voting: {'Name': 'Voting soft with grid', 'Single accuracy': 0.874, 'Cross accuracy': 0.85, 'STD': 0.021, 'Cross accuracy-STD*2': 0.808, 'Cross accuracy-STD*3': 0.788, 'Overfitting danger': 0.022, 'Time sec': 34}
	Stats Voting: {'Name': 'Voting hard with grid', 'Single accuracy': 0.891, 'Cross accuracy': 0.846, 'STD': 0.019, 'Cross accuracy-STD*2': 0.809, 'Cross accuracy-STD*3': 0.79, 'Overfitting danger': 0.04, 'Time sec': 38}
	Stats Emsemble: {'Name': 'Ensemble RF', 'Single accuracy': 0.901, 'Cross accuracy': 0.864, 'STD': 0.025, 'Cross accuracy-STD*2': 0.815, 'Cross accuracy-STD*3': 0.79, 'Overfitting danger': 0.033, 'Time sec': 15}
Conclusions: besides hard voting, and ensemble RF, everything is slightly better without most fares, and info about extremely large families (8+).  It's possible that now these ensembling less overfits

### 73. Return fare as continuous with Log (in addition to fare 13.5+) which for now will stay separate for Random Forest. Later perhaps need to remove
	Stats Single: {'Name': 'Log', 'Single accuracy': 0.846, 'Cross accuracy': 0.831, 'STD': 0.025, 'Cross accuracy-STD*2': 0.781, 'Cross accuracy-STD*3': 0.757, 'Overfitting danger': 0.013, 'Time sec': 0}
	Stats Single: {'Name': 'KNN 14', 'Single accuracy': 0.845, 'Cross accuracy': 0.825, 'STD': 0.03, 'Cross accuracy-STD*2': 0.766, 'Cross accuracy-STD*3': 0.736, 'Overfitting danger': 0.017, 'Time sec': 9}
	Stats Single: {'Name': 'SVM rbf', 'Single accuracy': 0.865, 'Cross accuracy': 0.841, 'STD': 0.014, 'Cross accuracy-STD*2': 0.812, 'Cross accuracy-STD*3': 0.798, 'Overfitting danger': 0.021, 'Time sec': 0}
	Stats Single: {'Name': 'SVM poly', 'Single accuracy': 0.877, 'Cross accuracy': 0.829, 'STD': 0.022, 'Cross accuracy-STD*2': 0.786, 'Cross accuracy-STD*3': 0.764, 'Overfitting danger': 0.041, 'Time sec': 0}
	Stats Single: {'Name': 'NB', 'Single accuracy': 0.788, 'Cross accuracy': 0.782, 'STD': 0.053, 'Cross accuracy-STD*2': 0.676, 'Cross accuracy-STD*3': 0.623, 'Overfitting danger': 0.004, 'Time sec': 0}
	Stats Single: {'Name': 'RF 8', 'Single accuracy': 0.907, 'Cross accuracy': 0.846, 'STD': 0.014, 'Cross accuracy-STD*2': 0.818, 'Cross accuracy-STD*3': 0.804, 'Overfitting danger': 0.055, 'Time sec': 8}
	Stats Single: {'Name': 'RF 7', 'Single accuracy': 0.898, 'Cross accuracy': 0.846, 'STD': 0.018, 'Cross accuracy-STD*2': 0.809, 'Cross accuracy-STD*3': 0.791, 'Overfitting danger': 0.046, 'Time sec': 7}
	Stats Single: {'Name': 'RF 6', 'Single accuracy': 0.883, 'Cross accuracy': 0.842, 'STD': 0.02, 'Cross accuracy-STD*2': 0.803, 'Cross accuracy-STD*3': 0.783, 'Overfitting danger': 0.037, 'Time sec': 8}
	Stats Single: {'Name': 'XGB', 'Single accuracy': 0.935, 'Cross accuracy': 0.849, 'STD': 0.012, 'Cross accuracy-STD*2': 0.824, 'Cross accuracy-STD*3': 0.812, 'Overfitting danger': 0.081, 'Time sec': 3}
	Stats Grid: {'Name': 'Grid KNN', 'Single accuracy': 0.847, 'Cross accuracy': 0.833, 'STD': 0.021, 'Cross accuracy-STD*2': 0.79, 'Cross accuracy-STD*3': 0.769, 'Overfitting danger': 0.012, 'Time sec': 11}
	Stats Grid: {'Name': 'Grid SVM', 'Single accuracy': 0.87, 'Cross accuracy': 0.844, 'STD': 0.013, 'Cross accuracy-STD*2': 0.819, 'Cross accuracy-STD*3': 0.806, 'Overfitting danger': 0.022, 'Time sec': 12}
	Stats Grid: {'Name': 'Grid RF', 'Single accuracy': 0.898, 'Cross accuracy': 0.846, 'STD': 0.018, 'Cross accuracy-STD*2': 0.809, 'Cross accuracy-STD*3': 0.791, 'Overfitting danger': 0.046, 'Time sec': 33}
	Stats Grid: {'Name': 'Grid XGB', 'Single accuracy': 0.935, 'Cross accuracy': 0.849, 'STD': 0.012, 'Cross accuracy-STD*2': 0.824, 'Cross accuracy-STD*3': 0.812, 'Overfitting danger': 0.081, 'Time sec': 15}
	Stats Voting: {'Name': 'Voting soft with grid', 'Single accuracy': 0.888, 'Cross accuracy': 0.845, 'STD': 0.019, 'Cross accuracy-STD*2': 0.807, 'Cross accuracy-STD*3': 0.788, 'Overfitting danger': 0.038, 'Time sec': 32}
	Stats Voting: {'Name': 'Voting hard with grid', 'Single accuracy': 0.902, 'Cross accuracy': 0.846, 'STD': 0.017, 'Cross accuracy-STD*2': 0.812, 'Cross accuracy-STD*3': 0.795, 'Overfitting danger': 0.051, 'Time sec': 31}
	Stats Emsemble: {'Name': 'Ensemble RF', 'Single accuracy': 0.956, 'Cross accuracy': 0.921, 'STD': 0.007, 'Cross accuracy-STD*2': 0.907, 'Cross accuracy-STD*3': 0.901, 'Overfitting danger': 0.033, 'Time sec': 9}
Conclusions:
	- Logistic got worse - makes sense since now we have continuous not linear new relationship.  Since connections here are not linear, we are OK with it.  Remove from hard voting
	- KNN - no major change
	- SVM rbf - got higher accuracy and lower STD - makes sense, like continuous variables
	- SVM poly - no majory change
	- NB - got even worse
	- RF 8 - got lower accuracy, but also lower STD, in general everything is better - makes sense now need a deeper tree with continuous fare variable
	- RF 7 - worse both accuracy and STD - consider not adding continuous fare to RF
	- RF 6 - got worse, makes sense, need more features due to continuous variable. Remove
	- XGB - got much better both accucary and STD, went up overfitting danger, so continue excluding for soft
	- Grid KNN - Got worse.  Wasn't great before also.
	- Grid SVM - got higher accuracy and lower STD - makes sense, like continuous variables
	- Grid RF (took depth 7) - worse accuracy and STD, consider not adding continuous fare to RF
	- XGB - got much better both accucary and STD, went up overfitting danger, so continue excluding for soft
	- Soft -  lower accuracy, but better STD, didn't change bottom line
	- Hard - got slighly better due to slightly lower STD
	- Ensemble - much better in everything
Bottom line:
	- Remove Log from Hard voting
	- Add RF9
	- Remove RF6
	- Don't add continuous fare to RF / and feature of 13.5+ fare to others? - one of next steps

### 74. See conclusions above
	Stats Single: {'Name': 'Log', 'Single accuracy': 0.846, 'Cross accuracy': 0.831, 'STD': 0.025, 'Cross accuracy-STD*2': 0.781, 'Cross accuracy-STD*3': 0.757, 'Overfitting danger': 0.013, 'Time sec': 0}
	Stats Single: {'Name': 'SVM rbf', 'Single accuracy': 0.865, 'Cross accuracy': 0.841, 'STD': 0.014, 'Cross accuracy-STD*2': 0.812, 'Cross accuracy-STD*3': 0.798, 'Overfitting danger': 0.021, 'Time sec': 0}
	Stats Single: {'Name': 'SVM poly', 'Single accuracy': 0.877, 'Cross accuracy': 0.829, 'STD': 0.022, 'Cross accuracy-STD*2': 0.786, 'Cross accuracy-STD*3': 0.764, 'Overfitting danger': 0.041, 'Time sec': 0}
	Stats Single: {'Name': 'NB', 'Single accuracy': 0.788, 'Cross accuracy': 0.782, 'STD': 0.053, 'Cross accuracy-STD*2': 0.676, 'Cross accuracy-STD*3': 0.623, 'Overfitting danger': 0.004, 'Time sec': 0}
	Stats Single: {'Name': 'RF 9', 'Single accuracy': 0.907, 'Cross accuracy': 0.846, 'STD': 0.014, 'Cross accuracy-STD*2': 0.818, 'Cross accuracy-STD*3': 0.804, 'Overfitting danger': 0.055, 'Time sec': 15}
	Stats Single: {'Name': 'RF 8', 'Single accuracy': 0.907, 'Cross accuracy': 0.846, 'STD': 0.014, 'Cross accuracy-STD*2': 0.818, 'Cross accuracy-STD*3': 0.804, 'Overfitting danger': 0.055, 'Time sec': 7}
	Stats Single: {'Name': 'RF 7', 'Single accuracy': 0.898, 'Cross accuracy': 0.846, 'STD': 0.018, 'Cross accuracy-STD*2': 0.809, 'Cross accuracy-STD*3': 0.791, 'Overfitting danger': 0.046, 'Time sec': 7}
	Stats Single: {'Name': 'XGB', 'Single accuracy': 0.935, 'Cross accuracy': 0.849, 'STD': 0.012, 'Cross accuracy-STD*2': 0.824, 'Cross accuracy-STD*3': 0.812, 'Overfitting danger': 0.081, 'Time sec': 3}
	Stats Grid: {'Name': 'Grid KNN', 'Single accuracy': 0.847, 'Cross accuracy': 0.833, 'STD': 0.021, 'Cross accuracy-STD*2': 0.79, 'Cross accuracy-STD*3': 0.769, 'Overfitting danger': 0.012, 'Time sec': 10}
	Stats Grid: {'Name': 'Grid SVM', 'Single accuracy': 0.87, 'Cross accuracy': 0.844, 'STD': 0.013, 'Cross accuracy-STD*2': 0.819, 'Cross accuracy-STD*3': 0.806, 'Overfitting danger': 0.022, 'Time sec': 10}
	Stats Grid: {'Name': 'Grid RF', 'Single accuracy': 0.898, 'Cross accuracy': 0.846, 'STD': 0.018, 'Cross accuracy-STD*2': 0.809, 'Cross accuracy-STD*3': 0.791, 'Overfitting danger': 0.046, 'Time sec': 30}
	Stats Grid: {'Name': 'Grid XGB', 'Single accuracy': 0.935, 'Cross accuracy': 0.849, 'STD': 0.012, 'Cross accuracy-STD*2': 0.824, 'Cross accuracy-STD*3': 0.812, 'Overfitting danger': 0.081, 'Time sec': 15}
	Stats Voting: {'Name': 'Voting soft with grid', 'Single accuracy': 0.889, 'Cross accuracy': 0.837, 'STD': 0.02, 'Cross accuracy-STD*2': 0.797, 'Cross accuracy-STD*3': 0.776, 'Overfitting danger': 0.046, 'Time sec': 31}
	Stats Voting: {'Name': 'Voting hard with grid', 'Single accuracy': 0.903, 'Cross accuracy': 0.849, 'STD': 0.015, 'Cross accuracy-STD*2': 0.819, 'Cross accuracy-STD*3': 0.804, 'Overfitting danger': 0.05, 'Time sec': 35}
	Stats Emsemble: {'Name': 'Ensemble RF', 'Single accuracy': 0.956, 'Cross accuracy': 0.919, 'STD': 0.009, 'Cross accuracy-STD*2': 0.901, 'Cross accuracy-STD*3': 0.893, 'Overfitting danger': 0.035, 'Time sec': 9}
Conclusions: 
	Soft voting got worse (because removed RF6 and added RF9 ? Not clear why)
	Hard voting got better in every way (because removed Logistic)
	Ensemble got slightly worse
	Remove special features that help RandomForests and other classificiations that prefer continuous features
Remove non-grid KNN since number of members changes every time we change features anyways

### 75. Split handling of Random Forest and another algorithms - currently only for 'Fare'
For Random Forest, leave fare binning, for others remove binning, and add continuous fare
	- Got better continuous: Log, SVM poly, NB
	- Got worse continuous: SVM rbf, Grid KNN 
	- Got better non-continuous (binned): RF7, Grid RF 
	- Didn't make a difference for: XGB, Grid XGB

	Stats Single: {'Name': 'Log', 'Single accuracy': 0.845, 'Cross accuracy': 0.831, 'STD': 0.022, 'Cross accuracy-STD*2': 0.786, 'Cross accuracy-STD*3': 0.763, 'Overfitting danger': 0.012, 'Time sec': 0}
	Stats Single: {'Name': 'SVM rbf', 'Single accuracy': 0.865, 'Cross accuracy': 0.84, 'STD': 0.016, 'Cross accuracy-STD*2': 0.808, 'Cross accuracy-STD*3': 0.793, 'Overfitting danger': 0.022, 'Time sec': 0}
	Stats Single: {'Name': 'SVM poly', 'Single accuracy': 0.88, 'Cross accuracy': 0.826, 'STD': 0.02, 'Cross accuracy-STD*2': 0.787, 'Cross accuracy-STD*3': 0.767, 'Overfitting danger': 0.047, 'Time sec': 0}
	Stats Single: {'Name': 'NB', 'Single accuracy': 0.805, 'Cross accuracy': 0.785, 'STD': 0.052, 'Cross accuracy-STD*2': 0.68, 'Cross accuracy-STD*3': 0.628, 'Overfitting danger': 0.016, 'Time sec': 0}
	Stats Single: {'Name': 'RF 9', 'Single accuracy': 0.895, 'Cross accuracy': 0.849, 'STD': 0.018, 'Cross accuracy-STD*2': 0.813, 'Cross accuracy-STD*3': 0.796, 'Overfitting danger': 0.041, 'Time sec': 17}
	Stats Single: {'Name': 'RF 8', 'Single accuracy': 0.89, 'Cross accuracy': 0.85, 'STD': 0.018, 'Cross accuracy-STD*2': 0.814, 'Cross accuracy-STD*3': 0.797, 'Overfitting danger': 0.036, 'Time sec': 7}
	Stats Single: {'Name': 'RF 7', 'Single accuracy': 0.888, 'Cross accuracy': 0.851, 'STD': 0.015, 'Cross accuracy-STD*2': 0.82, 'Cross accuracy-STD*3': 0.805, 'Overfitting danger': 0.033, 'Time sec': 7}
	Stats Single: {'Name': 'XGB', 'Single accuracy': 0.935, 'Cross accuracy': 0.849, 'STD': 0.012, 'Cross accuracy-STD*2': 0.824, 'Cross accuracy-STD*3': 0.812, 'Overfitting danger': 0.081, 'Time sec': 2}
	Stats Grid: {'Name': 'Grid KNN', 'Single accuracy': 0.844, 'Cross accuracy': 0.833, 'STD': 0.023, 'Cross accuracy-STD*2': 0.787, 'Cross accuracy-STD*3': 0.765, 'Overfitting danger': 0.009, 'Time sec': 9}
	Stats Grid: {'Name': 'Grid SVM', 'Single accuracy': 0.884, 'Cross accuracy': 0.844, 'STD': 0.022, 'Cross accuracy-STD*2': 0.801, 'Cross accuracy-STD*3': 0.779, 'Overfitting danger': 0.036, 'Time sec': 8}
	Stats Grid: {'Name': 'Grid RF', 'Single accuracy': 0.877, 'Cross accuracy': 0.851, 'STD': 0.017, 'Cross accuracy-STD*2': 0.816, 'Cross accuracy-STD*3': 0.799, 'Overfitting danger': 0.023, 'Time sec': 24}
	Stats Grid: {'Name': 'Grid XGB', 'Single accuracy': 0.935, 'Cross accuracy': 0.849, 'STD': 0.012, 'Cross accuracy-STD*2': 0.824, 'Cross accuracy-STD*3': 0.812, 'Overfitting danger': 0.081, 'Time sec': 11}
- Re-add binned strong features for continuous classifiers, but keep Random Forest without continuous fare feature
- Can't do soft and hard voting for continuous and not continous classifiers together since they have a different number of features, so leaving only for continous
- Ensembling doing for all together
	Stats Single: {'Name': 'Log', 'Single accuracy': 0.847, 'Cross accuracy': 0.828, 'STD': 0.025, 'Cross accuracy-STD*2': 0.778, 'Cross accuracy-STD*3': 0.753, 'Overfitting danger': 0.016, 'Time sec': 0}
	Stats Single: {'Name': 'SVM rbf', 'Single accuracy': 0.866, 'Cross accuracy': 0.841, 'STD': 0.014, 'Cross accuracy-STD*2': 0.812, 'Cross accuracy-STD*3': 0.798, 'Overfitting danger': 0.022, 'Time sec': 0}
	Stats Single: {'Name': 'SVM poly', 'Single accuracy': 0.878, 'Cross accuracy': 0.825, 'STD': 0.024, 'Cross accuracy-STD*2': 0.778, 'Cross accuracy-STD*3': 0.754, 'Overfitting danger': 0.046, 'Time sec': 0}
	Stats Single: {'Name': 'NB', 'Single accuracy': 0.788, 'Cross accuracy': 0.782, 'STD': 0.053, 'Cross accuracy-STD*2': 0.676, 'Cross accuracy-STD*3': 0.623, 'Overfitting danger': 0.004, 'Time sec': 0}
	Stats Single: {'Name': 'RF 9', 'Single accuracy': 0.895, 'Cross accuracy': 0.847, 'STD': 0.019, 'Cross accuracy-STD*2': 0.809, 'Cross accuracy-STD*3': 0.789, 'Overfitting danger': 0.042, 'Time sec': 20}
	Stats Single: {'Name': 'RF 8', 'Single accuracy': 0.89, 'Cross accuracy': 0.85, 'STD': 0.018, 'Cross accuracy-STD*2': 0.814, 'Cross accuracy-STD*3': 0.797, 'Overfitting danger': 0.036, 'Time sec': 8}
	Stats Single: {'Name': 'RF 7', 'Single accuracy': 0.888, 'Cross accuracy': 0.852, 'STD': 0.017, 'Cross accuracy-STD*2': 0.817, 'Cross accuracy-STD*3': 0.8, 'Overfitting danger': 0.032, 'Time sec': 8}
	Stats Single: {'Name': 'XGB', 'Single accuracy': 0.936, 'Cross accuracy': 0.847, 'STD': 0.014, 'Cross accuracy-STD*2': 0.819, 'Cross accuracy-STD*3': 0.804, 'Overfitting danger': 0.083, 'Time sec': 3}
	Stats Grid: {'Name': 'Grid KNN', 'Single accuracy': 0.848, 'Cross accuracy': 0.833, 'STD': 0.021, 'Cross accuracy-STD*2': 0.79, 'Cross accuracy-STD*3': 0.769, 'Overfitting danger': 0.013, 'Time sec': 11}
	Stats Grid: {'Name': 'Grid SVM', 'Single accuracy': 0.87, 'Cross accuracy': 0.845, 'STD': 0.014, 'Cross accuracy-STD*2': 0.817, 'Cross accuracy-STD*3': 0.803, 'Overfitting danger': 0.021, 'Time sec': 11}
	Stats Grid: {'Name': 'Grid RF', 'Single accuracy': 0.877, 'Cross accuracy': 0.853, 'STD': 0.019, 'Cross accuracy-STD*2': 0.814, 'Cross accuracy-STD*3': 0.795, 'Overfitting danger': 0.021, 'Time sec': 33}
	Stats Grid: {'Name': 'Grid XGB', 'Single accuracy': 0.936, 'Cross accuracy': 0.847, 'STD': 0.014, 'Cross accuracy-STD*2': 0.819, 'Cross accuracy-STD*3': 0.804, 'Overfitting danger': 0.083, 'Time sec': 15}
	Stats Voting: {'Name': 'Voting soft with grid', 'Single accuracy': 0.862, 'Cross accuracy': 0.844, 'STD': 0.018, 'Cross accuracy-STD*2': 0.809, 'Cross accuracy-STD*3': 0.791, 'Overfitting danger': 0.015, 'Time sec': 1}
	Stats Voting: {'Name': 'Voting hard with grid', 'Single accuracy': 0.899, 'Cross accuracy': 0.849, 'STD': 0.016, 'Cross accuracy-STD*2': 0.817, 'Cross accuracy-STD*3': 0.802, 'Overfitting danger': 0.045, 'Time sec': 6}
	Stats Emsemble: {'Name': 'Ensemble RF', 'Single accuracy': 0.956, 'Cross accuracy': 0.916, 'STD': 0.01, 'Cross accuracy-STD*2': 0.896, 'Cross accuracy-STD*3': 0.886, 'Overfitting danger': 0.039, 'Time sec': 9}
- Ensemble RF - clearly best, but clearly overfits
- SVM, XGB give great results
- RF7 no Grid gives good results
- Voting hard gives good results - high correlation with XGB, and SVM
- Voting soft doesn't give great results - high correlation with SVM

### 76. Removing a few categories of age that I assume don't help much for Random Forest (before adding continous age) as benchmark
	Stats Single: {'Name': 'Log', 'Single accuracy': 0.844, 'Cross accuracy': 0.832, 'STD': 0.027, 'Cross accuracy-STD*2': 0.777, 'Cross accuracy-STD*3': 0.75, 'Overfitting danger': 0.01, 'Time sec': 0}
	Stats Single: {'Name': 'SVM rbf', 'Single accuracy': 0.854, 'Cross accuracy': 0.834, 'STD': 0.021, 'Cross accuracy-STD*2': 0.793, 'Cross accuracy-STD*3': 0.772, 'Overfitting danger': 0.017, 'Time sec': 0}
	Stats Single: {'Name': 'SVM poly', 'Single accuracy': 0.861, 'Cross accuracy': 0.816, 'STD': 0.032, 'Cross accuracy-STD*2': 0.751, 'Cross accuracy-STD*3': 0.719, 'Overfitting danger': 0.039, 'Time sec': 0}
	Stats Single: {'Name': 'NB', 'Single accuracy': 0.786, 'Cross accuracy': 0.784, 'STD': 0.054, 'Cross accuracy-STD*2': 0.675, 'Cross accuracy-STD*3': 0.621, 'Overfitting danger': 0.002, 'Time sec': 0}
	Stats Single: {'Name': 'RF 9', 'Single accuracy': 0.88, 'Cross accuracy': 0.834, 'STD': 0.024, 'Cross accuracy-STD*2': 0.786, 'Cross accuracy-STD*3': 0.762, 'Overfitting danger': 0.04, 'Time sec': 18}
	Stats Single: {'Name': 'RF 8', 'Single accuracy': 0.878, 'Cross accuracy': 0.84, 'STD': 0.021, 'Cross accuracy-STD*2': 0.798, 'Cross accuracy-STD*3': 0.778, 'Overfitting danger': 0.033, 'Time sec': 8}
	Stats Single: {'Name': 'RF 7', 'Single accuracy': 0.873, 'Cross accuracy': 0.838, 'STD': 0.018, 'Cross accuracy-STD*2': 0.802, 'Cross accuracy-STD*3': 0.783, 'Overfitting danger': 0.03, 'Time sec': 9}
	Stats Single: {'Name': 'XGB', 'Single accuracy': 0.927, 'Cross accuracy': 0.824, 'STD': 0.021, 'Cross accuracy-STD*2': 0.781, 'Cross accuracy-STD*3': 0.76, 'Overfitting danger': 0.096, 'Time sec': 3}
	Stats Grid: {'Name': 'Grid KNN', 'Single accuracy': 0.845, 'Cross accuracy': 0.832, 'STD': 0.019, 'Cross accuracy-STD*2': 0.795, 'Cross accuracy-STD*3': 0.776, 'Overfitting danger': 0.011, 'Time sec': 12}
	Stats Grid: {'Name': 'Grid SVM', 'Single accuracy': 0.854, 'Cross accuracy': 0.838, 'STD': 0.021, 'Cross accuracy-STD*2': 0.797, 'Cross accuracy-STD*3': 0.776, 'Overfitting danger': 0.013, 'Time sec': 11}
	Stats Grid: {'Name': 'Grid RF', 'Single accuracy': 0.855, 'Cross accuracy': 0.842, 'STD': 0.017, 'Cross accuracy-STD*2': 0.808, 'Cross accuracy-STD*3': 0.791, 'Overfitting danger': 0.011, 'Time sec': 33}
	Stats Grid: {'Name': 'Grid XGB', 'Single accuracy': 0.903, 'Cross accuracy': 0.845, 'STD': 0.019, 'Cross accuracy-STD*2': 0.808, 'Cross accuracy-STD*3': 0.79, 'Overfitting danger': 0.053, 'Time sec': 13}
	Stats Voting: {'Name': 'Voting soft with grid', 'Single accuracy': 0.848, 'Cross accuracy': 0.828, 'STD': 0.025, 'Cross accuracy-STD*2': 0.779, 'Cross accuracy-STD*3': 0.754, 'Overfitting danger': 0.017, 'Time sec': 1}
	Stats Voting: {'Name': 'Voting hard with grid', 'Single accuracy': 0.884, 'Cross accuracy': 0.838, 'STD': 0.011, 'Cross accuracy-STD*2': 0.816, 'Cross accuracy-STD*3': 0.805, 'Overfitting danger': 0.041, 'Time sec': 5}
	Stats Emsemble: {'Name': 'Ensemble RF', 'Single accuracy': 0.945, 'Cross accuracy': 0.9, 'STD': 0.011, 'Cross accuracy-STD*2': 0.877, 'Cross accuracy-STD*3': 0.866, 'Overfitting danger': 0.042, 'Time sec': 10}
Conclusion:before readding full continous edge - benchmark:
- SVM got a big hit (but that's probably good because it was overfitting on these age features)
- XGB got a big hit (was probably also overfitting on these features)
- RFs actually improved - that's a good sign, since it strengthens the fact that these age features were not really helping

### 77. Making age continous (in addition to <4) 
	- SVM got drastically worse (accuracy and STD)
	- RF7 got slightly better, RF8 & 9 got much better. RF8 is the best. Makes sense sense need larger depth
	- XGB got much better (but probably overfitting)
	- KNN got much better 
	- Grid RF got better (smaller STD)
	
	Stats Single: {'Name': 'Log', 'Single accuracy': 0.841, 'Cross accuracy': 0.84, 'STD': 0.025, 'Cross accuracy-STD*2': 0.79, 'Cross accuracy-STD*3': 0.765, 'Overfitting danger': 0.001, 'Time sec': 0}
	Stats Single: {'Name': 'SVM rbf', 'Single accuracy': 0.857, 'Cross accuracy': 0.836, 'STD': 0.019, 'Cross accuracy-STD*2': 0.798, 'Cross accuracy-STD*3': 0.78, 'Overfitting danger': 0.018, 'Time sec': 0}
	Stats Single: {'Name': 'SVM poly', 'Single accuracy': 0.872, 'Cross accuracy': 0.83, 'STD': 0.025, 'Cross accuracy-STD*2': 0.78, 'Cross accuracy-STD*3': 0.755, 'Overfitting danger': 0.037, 'Time sec': 0}
	Stats Single: {'Name': 'NB', 'Single accuracy': 0.793, 'Cross accuracy': 0.785, 'STD': 0.055, 'Cross accuracy-STD*2': 0.675, 'Cross accuracy-STD*3': 0.62, 'Overfitting danger': 0.007, 'Time sec': 0}
	Stats Single: {'Name': 'RF 9', 'Single accuracy': 0.924, 'Cross accuracy': 0.844, 'STD': 0.017, 'Cross accuracy-STD*2': 0.811, 'Cross accuracy-STD*3': 0.794, 'Overfitting danger': 0.074, 'Time sec': 17}
	Stats Single: {'Name': 'RF 8', 'Single accuracy': 0.914, 'Cross accuracy': 0.85, 'STD': 0.015, 'Cross accuracy-STD*2': 0.82, 'Cross accuracy-STD*3': 0.805, 'Overfitting danger': 0.058, 'Time sec': 7}
	Stats Single: {'Name': 'RF 7', 'Single accuracy': 0.897, 'Cross accuracy': 0.851, 'STD': 0.016, 'Cross accuracy-STD*2': 0.818, 'Cross accuracy-STD*3': 0.802, 'Overfitting danger': 0.041, 'Time sec': 7}
	Stats Single: {'Name': 'XGB', 'Single accuracy': 0.962, 'Cross accuracy': 0.85, 'STD': 0.008, 'Cross accuracy-STD*2': 0.833, 'Cross accuracy-STD*3': 0.825, 'Overfitting danger': 0.108, 'Time sec': 2}
	Stats Grid: {'Name': 'Grid KNN', 'Single accuracy': 0.863, 'Cross accuracy': 0.841, 'STD': 0.018, 'Cross accuracy-STD*2': 0.806, 'Cross accuracy-STD*3': 0.788, 'Overfitting danger': 0.019, 'Time sec': 10}
	Stats Grid: {'Name': 'Grid SVM', 'Single accuracy': 0.868, 'Cross accuracy': 0.842, 'STD': 0.017, 'Cross accuracy-STD*2': 0.807, 'Cross accuracy-STD*3': 0.79, 'Overfitting danger': 0.022, 'Time sec': 11}
	Stats Grid: {'Name': 'Grid RF', 'Single accuracy': 0.897, 'Cross accuracy': 0.851, 'STD': 0.016, 'Cross accuracy-STD*2': 0.818, 'Cross accuracy-STD*3': 0.802, 'Overfitting danger': 0.041, 'Time sec': 30}
	Stats Grid: {'Name': 'Grid XGB', 'Single accuracy': 0.93, 'Cross accuracy': 0.86, 'STD': 0.018, 'Cross accuracy-STD*2': 0.824, 'Cross accuracy-STD*3': 0.806, 'Overfitting danger': 0.066, 'Time sec': 13}
	Stats Voting: {'Name': 'Voting soft with grid', 'Single accuracy': 0.857, 'Cross accuracy': 0.837, 'STD': 0.02, 'Cross accuracy-STD*2': 0.798, 'Cross accuracy-STD*3': 0.778, 'Overfitting danger': 0.017, 'Time sec': 1}
	Stats Voting: {'Name': 'Voting hard with grid', 'Single accuracy': 0.899, 'Cross accuracy': 0.859, 'STD': 0.019, 'Cross accuracy-STD*2': 0.822, 'Cross accuracy-STD*3': 0.803, 'Overfitting danger': 0.036, 'Time sec': 5}
	Stats Emsemble: {'Name': 'Ensemble RF', 'Single accuracy': 0.989, 'Cross accuracy': 0.969, 'STD': 0.013, 'Cross accuracy-STD*2': 0.942, 'Cross accuracy-STD*3': 0.928, 'Overfitting danger': 0.02, 'Time sec': 8}


