import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("adult.csv", header = None)
df.columns = ["Age", "Workclass","Fnlwgt","Education","Education-num","Marital-status","Occupation", "Relationship","Race","Sex","capital-gain","capital-loss","hours-per-week","native-country","income"]

label_encoder = LabelEncoder()
categorical_columns = ["Workclass", "Education", "Marital-status", "Occupation", "Relationship", "Race", "Sex", "native-country", "income"]

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Init
gaussian_nb = GaussianNB()
multinomial_nb = MultinomialNB()
bernoulli_nb = BernoulliNB()

# Train 
gaussian_nb.fit(X_train, y_train)
multinomial_nb.fit(X_train, y_train)
bernoulli_nb.fit(X_train, y_train)

#predic
gaussian_predictions = gaussian_nb.predict(X_test)
multinomial_predictions = multinomial_nb.predict(X_test)
bernoulli_predictions = bernoulli_nb.predict(X_test)

def evaluate_model(predictions, model_name):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"{model_name} Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print()

evaluate_model(gaussian_predictions, "GaussianNB")
evaluate_model(multinomial_predictions, "MultinomialNB")
evaluate_model(bernoulli_predictions, "BernoulliNB")