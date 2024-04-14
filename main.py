import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer

# Load the CC dataset
cc_data = pd.read_csv('/content/CC GENERAL.csv')

# Exclude non-numeric columns before scaling
numeric_columns = cc_data.select_dtypes(include=['number']).columns
cc_data_numeric = cc_data[numeric_columns]

# Impute missing values for CC dataset
imputer_cc = SimpleImputer(strategy='mean')
imputed_data_cc = imputer_cc.fit_transform(cc_data_numeric)

# Perform Scaling for CC dataset
scaler_cc = StandardScaler()
scaled_data_cc = scaler_cc.fit_transform(imputed_data_cc)

# Apply PCA for CC dataset
pca_cc = PCA(n_components=2)
pca_result_cc = pca_cc.fit_transform(scaled_data_cc)

# Apply k-means algorithm on PCA result for CC dataset
kmeans_cc = KMeans(n_clusters=3)
kmeans_labels_cc = kmeans_cc.fit_predict(pca_result_cc)

# Calculate silhouette score on scaled PCA result for CC dataset
silhouette_avg_cc = silhouette_score(pca_result_cc, kmeans_labels_cc)
print("Silhouette Score for CC dataset:", silhouette_avg_cc)

# Load pd_speech_features.csv
speech_data = pd.read_csv('/content/pd_speech_features.csv')

# Handle missing values for speech data
imputer_speech = SimpleImputer(strategy='mean')
imputed_data_speech = imputer_speech.fit_transform(speech_data)

# Assuming the target column is named 'class' instead of 'target'
X_speech = pd.DataFrame(imputed_data_speech, columns=speech_data.columns)
X_speech = X_speech.drop('class', axis=1)  # Assuming 'class' is the target column
y_speech = speech_data['class']

# Perform Scaling for speech data
scaler_speech = StandardScaler()
scaled_X_speech = scaler_speech.fit_transform(X_speech)

# Apply PCA (k=3) for speech data
pca_speech = PCA(n_components=3)
pca_result_speech = pca_speech.fit_transform(scaled_X_speech)

# Split data for SVM on speech data
X_train_speech, X_test_speech, y_train_speech, y_test_speech = train_test_split(pca_result_speech, y_speech, test_size=0.2, random_state=42)

# Use SVM to report performance for speech data
svm_model_speech = SVC()
svm_model_speech.fit(X_train_speech, y_train_speech)
y_pred_speech = svm_model_speech.predict(X_test_speech)

accuracy_speech = accuracy_score(y_test_speech, y_pred_speech)
print("SVM Accuracy for pd_speech_features.csv dataset:", accuracy_speech)

# Load Iris dataset
iris_data = pd.read_csv('/content/Iris.csv')

# Check the column names and adjust accordingly
print(iris_data.columns)

# Assuming the target column is named 'Species' instead of 'class'
X_iris = iris_data.drop('Species', axis=1)
y_iris = iris_data['Species']

# Perform Scaling for Iris dataset
scaler_iris = StandardScaler()
scaled_X_iris = scaler_iris.fit_transform(X_iris)

# Apply LDA (k=2) for Iris dataset
lda = LDA(n_components=2)
lda_result = lda.fit_transform(scaled_X_iris, y_iris)

# Display LDA result for Iris dataset
print("LDA Result for Iris dataset:")
print(lda_result)
