
#delete this part in the report
time_list = []
split_AUC = []

for i in range(1,6):
    #the actual prediction model section (tested in a different file and it passes all 5 splits)
    #Best Hyperparameters: {'C': 0.01, 'l1_ratio': 0.1}
    #AUC Score: 0.98693
    
    start_time = time.time()

    folder_name = f'F24_Proj3_data/split_{i}'
    train_file = os.path.join(folder_name, 'train.csv')
    test_file = os.path.join(folder_name, 'test.csv')
    test_labels_file = os.path.join(folder_name, 'test_y.csv')

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    test_labels = pd.read_csv(test_labels_file)

    X_train = train_data.iloc[:, 3:].values  
    y_train = train_data['sentiment'].values
    X_test = test_data.iloc[:, 2:].values  
    y_test = test_labels['sentiment'].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pred_model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=500, C=0.01, l1_ratio=0.1)
    pred_model.fit(X_train, y_train)

    y_pred_proba = pred_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    split_AUC.append(auc_score)
    
    end_time = time.time()
    execution_time = end_time-start_time
    time_list.append(execution_time)