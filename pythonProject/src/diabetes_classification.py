import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('../data/diabetes.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

print(df)

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=125)

# Skalowanie danych dla k-NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


results = []
conf_matrices = {}

## Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)
results.append({'Classifier': 'Naive Bayes', 'Accuracy': nb_accuracy * 100})
conf_matrices['Naive Bayes'] = cm_nb

print("Naive Bayes:")
print(f"Dokładność: {nb_accuracy:.2%}")
print("Macierz błędu:")
print(cm_nb)
print("\n" + "=" * 50 + "\n")

##k-NN
k_values = [3, 5, 11]
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    results.append({'Classifier': f'k-NN (k={k})', 'Accuracy': knn_accuracy * 100})
    conf_matrices[f'k-NN (k={k})'] = cm_knn

    print(f"k-NN (k={k}):")
    print(f"Dokładność: {knn_accuracy:.2%}")
    print("Macierz błędu:")
    print(cm_knn)
    print("\n" + "=" * 50 + "\n")



# Przygotowanie danych do wykresu
results_df = pd.DataFrame(results)

# Tworzenie wykresu
plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    x='Classifier',
    y='Accuracy',
    hue='Classifier',
    data=results_df,
    palette='viridis',
    legend=False
)

# Dodanie wartości na słupkach
for p in barplot.patches:
    barplot.annotate(
        f'{p.get_height():.1f}%',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center',
        va='center',
        xytext=(0, 10),
        textcoords='offset points',
        fontsize=10
    )

# Konfiguracja wykresu
plt.title('Porównanie dokładności klasyfikatorów', fontsize=14)
plt.xlabel('Klasyfikator', fontsize=12)
plt.ylabel('Dokładność (%)', fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

