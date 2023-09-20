import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import eli5
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


def do_my_study():
    df = get_data_and_preproc()

    df.to_csv('preproc_diabetic_data.csv') 

    # cat_and_num = clarify_features()

    # result = preproc(df, cat_and_num[0], cat_and_num[1])

    # # logreg = find_score_and_best_NONSMOTE(
    # logreg = find_score_SMOTE(
    #     result[0], result[1], result[2], result[3],
    #     result[4], result[5], result[6])

    # find_important(logreg, cat_and_num[0], cat_and_num[1])

    # Sanity check
    # print(df.isnull().sum())


def get_data_and_preproc():
    df = pd.read_csv("diabetic_data.csv")

    # Replacing '?' values with NULL
    df = df.replace('?', np.nan)

    # Removed Features:
    # encounter_id and patient_nbr are unique IDs
    # weight has 97% missing
    # payer_code and medical_specialty both have 40% and 49% missing
    # examide and citoglipton both have only 1 value
    df = df.drop(columns=['encounter_id', 'patient_nbr', 'weight',
                          'payer_code', 'medical_specialty', 'examide',
                          'citoglipton'])

    df['race'] = df['race'].fillna(
        pd.Series(
            np.random.choice(
                ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Other', 'Asian'],
                p=[0.7106, 0.1889, 0.0225, 0.0148, 0.0632],
                size=len(df))))

    df["diag_1"] = df["diag_1"].apply(categorize_diag_code)
    df["diag_2"] = df["diag_2"].apply(categorize_diag_code)
    df["diag_3"] = df["diag_3"].apply(categorize_diag_code)

    # In column admission_type_id,
    # according to IDs_mapping.csv, 5 (Not Available) and 8 (Not Mapped)
    # can be moerged into 6 (NULL)
    df.admission_type_id.replace(
        to_replace={5, 8}, value=6, inplace=True)
    df.admission_type_id = df.admission_type_id.astype(str)

    # In admission_source_id,
    # according to IDs_mapping.csv, 9 and 15 (Not Available) and
    #                     20 (Not Mapped) and 21 (Unknown/Invalid)
    # can be moerged into 6 (NULL)
    df.admission_source_id.replace(
        to_replace={9, 15, 20, 21}, value=17, inplace=True)
    df.admission_source_id = df.admission_source_id.astype(str)

    # Removing rows of discharge_disposition_id,
    # where it indicates non-readmission
    remove = df[df['discharge_disposition_id'].isin(
        [11, 13, 14, 19, 20, 21])].index
    df.drop(remove, inplace=True)

    # Convert response variable into binary
    # If readmitted within (less than) 30 days, value = 1, otherwise value = 0
    binary = {'NO': 0, '>30': 0, '<30': 1}
    df["readmitted"].replace(binary, inplace=True)

    # Prepare admission_type_id for visualization
    di = {'1': 'Emergency', '2': 'Urgent',
          '3': 'Elective', '4': 'Newborn',
          '6': 'Not Available', '7': 'Trauma Center'}
    df.admission_type_id.replace(di, inplace=True)

    di = {25: 'Not Available', 1: 'Home',
          3: 'SNF', 6: 'Home Health',
          2: 'Short Term Hospital', 5: 'Inpatient Care Institution',
          7: 'Left AMA', 10: 'Neonatal Aftercare',
          4: 'ICF', 18: 'Not Available',
          8: 'Home IV', 12: 'Still Patient',
          16: 'Another Institution', 17: 'This Institution',
          22: 'Rehab', 23: 'Long Term Hospital',
          9: 'Admitted', 15: 'Swing Bed',
          24: 'Nursing Facility Under Medicaid Not Medicare',
          28: 'Psychiatric Hospital', 27: 'Federal Health Care Facility'}
    df.discharge_disposition_id.replace(di, inplace=True)

    di = {'1': 'Physician Referral', '7': 'Emergency Room',
          '2': 'Clinic Referral', '4': 'Transfer from a hospital',
          '5': 'Transfer from a Skilled Nursing Facility (SNF)', '17': 'Not Available',
          '6': 'Transfer from another health care facility', '3': 'HMO Referral',
          '8': 'Court/Law Enforcement', '14': 'Extramural Birth',
          '10': 'Transfer from critial access hospital', '22': 'Transfer from hospital inpt/same fac reslt in a sep claim',
          '11': 'Normal Delivery', '25': 'Transfer from Ambulatory Surgery Center', '13': 'Sick Baby'}
    df.admission_source_id.replace(di, inplace=True)

    return df


def categorize_diag_code(code):
    try:
        code = float(code)
    except ValueError:
        code = 0

    # Circulatory
    if code in range(390, 460) or code == 785:
        return ("Circulatory")

    # Respiratory
    elif code in range(460, 520) or code == 786:
        return ("Respiratory")

    # Digestive
    elif code in range(520, 580) or code == 787:
        return ("Digestive")

    # Diabetes
    elif code >= 250 and code < 251:
        return ("Diabetes")

    # Injury
    elif code in range(800, 1000):
        return ("Injury")

    # Musculoskeletal
    elif code in range(710, 740):
        return ("Musculoskeletal")

    # Genitourinary
    elif code in range(580, 630) or code == 788:
        return ("Genitourinary")

    # Neoplasms
    elif code == 784 or code in (range(140, 240) or range(780, 783) or
                                 range(790, 800) or range(240, 250) or
                                 range(251, 280) or range(680, 710) or
                                 range(1, 140) or range(290, 320)):
        return ("Neoplasms")

    # Other
    else:
        return ("Other")


def clarify_features():
    categorical_features = ['race', 'gender', 'max_glu_serum', 'A1Cresult',
                            'metformin', 'repaglinide', 'nateglinide',
                            'chlorpropamide', 'glimepiride', 'acetohexamide',
                            'glipizide', 'glyburide', 'tolbutamide',
                            'pioglitazone', 'rosiglitazone', 'acarbose',
                            'miglitol', 'troglitazone', 'tolazamide',
                            'insulin', 'glyburide-metformin',
                            'glipizide-metformin', 'age',
                            'glimepiride-pioglitazone',
                            'metformin-rosiglitazone',
                            'metformin-pioglitazone', 'change', 'diabetesMed',
                            'diag_1', 'diag_2', 'diag_3',
                            'admission_type_id', 'discharge_disposition_id',
                            'admission_source_id']

    numeric_features = ['time_in_hospital', 'num_lab_procedures',
                        'num_procedures', 'num_medications',
                        'number_outpatient', 'number_emergency',
                        'number_inpatient', 'number_diagnoses']

    return categorical_features, numeric_features


# random_state = 11
def preproc(df, categorical_features, numeric_features):
    X = df[categorical_features + numeric_features]
    Y = df['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, stratify=Y, random_state=11)

    warnings.filterwarnings('ignore')

    numeric_transformer = RobustScaler(with_centering=False)
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numeric_features)])

    stratified_kfold = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=11)

    # General strengths range
    # param_grid = [{'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],

    # Best Non-SMOTE
    # param_grid = [{'classifier__C': [0.09,
    #                                  0.091, 0.092, 0.093, 0.094, 0.095,
    #                                  0.096, 0.097, 0.098, 0.099,
    #                                  0.1,
    #                                  0.11, 0.12, 0.13, 0.14, 0.15,
    #                                  0.16, 0.17, 0.18, 0.19,
    #                                  0.2],

    # Best SMOTE
    param_grid = [{'classifier__C': [0.001,
                                     0.002, 0.003, 0.004, 0.005,
                                     0.006, 0.007, 0.008, 0.009,
                                     0.01,
                                     0.02, 0.03, 0.04, 0.05,
                                     0.06, 0.07, 0.08, 0.09,
                                     0.1],
                   'classifier__penalty': ['l1'],  # switch to L2 if want
                   'classifier__solver': ['saga']}]

    return (X_train, X_test, y_train, y_test,
            preprocessor, stratified_kfold, param_grid)


def find_score_NONSMOTE(X_train, X_test, y_train, y_test,
                        preprocessor, stratified_kfold, param_grid):
    pipeline = imbpipeline(steps=[
        ['preprocessor', preprocessor],
        ['classifier', LogisticRegression(
            random_state=11, max_iter=1000, n_jobs=-1)]])

    logreg = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, scoring='roc_auc',
        cv=stratified_kfold, n_jobs=-1)

    logreg.fit(X_train, y_train)
    test_score = logreg.score(X_test, y_test)

    print(f'Non-SMOTE Cross-validation score: \t{logreg.best_score_}',
          f'\nNon-SMOTE Test score: \t\t{test_score}')

    print(logreg.best_params_)


def find_score_SMOTE(X_train, X_test, y_train, y_test,
                     preprocessor, stratified_kfold, param_grid):
    pipeline = imbpipeline(steps=[
        ['preprocessor', preprocessor],
        ['smote', SMOTE(random_state=11)],
        ['classifier', LogisticRegression(
            random_state=11, max_iter=1000, n_jobs=-1)]])

    logreg = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, scoring='roc_auc',
        cv=stratified_kfold, n_jobs=-1)

    logreg.fit(X_train, y_train)
    test_score = logreg.score(X_test, y_test)

    print(f'SMOTE Cross-validation score: \t{logreg.best_score_}',
          f'\nSMOTE Test score: \t\t{test_score}')

    print(logreg.best_params_)

    threshold = 0.48
    y_pred = (
        logreg.predict_proba(X_test)[:, 1] > threshold).astype('float')
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, cmap='Blues', normalize='true')

    return logreg


def find_important(logreg, categorical_features, numeric_features):

    onehot_columns = list(
        logreg.best_estimator_.named_steps['preprocessor'].
        named_transformers_['cat'].
        get_feature_names(input_features=categorical_features))
    numeric_features_list = list(numeric_features)
    numeric_features_list.extend(onehot_columns)

    print(eli5.format_as_dataframe(eli5.explain_weights(
        logreg.best_estimator_.named_steps['classifier'],
        top=23, feature_names=numeric_features_list)).head(10))


if __name__ == "__main__":

    do_my_study()
