disp('PREPROCESSING');
run('preprocessing.m');

disp('TRAINING DATASET CREATION');
run('svm_train_dataset.m');

disp('TRAINING');
run('svm_training.m');

disp('TEST DATASET CREATION');
run('svm_test_dataset.m');

disp('TESTING');
run('svm_testing.m');