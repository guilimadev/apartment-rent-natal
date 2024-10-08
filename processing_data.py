import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from scrapper import builder


df_apartments = builder()
df_apartments = df_apartments[~df_apartments['Aluguel'].str.contains('/dia')]
df_apartments['Aluguel'] = df_apartments['Aluguel'].replace(r'[^\d]', '', regex=True).astype(float)


# Keep a copy of the original 'Bairro' column for filtering
bairros = df_apartments['Bairro'].copy()

# Create a copy of the original data before normalization
df_apartments_original = df_apartments.copy() 
df_apartments = pd.get_dummies(df_apartments, columns=['Bairro'])

# Convert columns to numeric
df_apartments['Aluguel'] = pd.to_numeric(df_apartments['Aluguel'], errors='coerce')
df_apartments['Tamanho'] = pd.to_numeric(df_apartments['Tamanho'], errors='coerce')
df_apartments['Quartos'] = df_apartments['Quartos'].str.extract('(\d+)').astype(float)
df_apartments['Banheiros'] = df_apartments['Banheiros'].str.extract('(\d+)').astype(float)

df_apartments['Aluguel Total'] = df_apartments['Aluguel'] + df_apartments['Extras']

# Drop rows with missing 'Aluguel' or 'Tamanho'
df_apartments.dropna(subset=['Aluguel', 'Tamanho'], inplace=True)

# Calculate price per square meter
df_apartments['preco_por_m2'] = df_apartments['Aluguel Total'] / df_apartments['Tamanho']

df_apartments.drop(columns=['Localização'], inplace=True)

# Normalization
scaler = StandardScaler()
df_apartments[['Aluguel Total', 'Tamanho', 'preco_por_m2', 'Quartos', 'Banheiros']] = scaler.fit_transform(df_apartments[['Aluguel Total', 'Tamanho', 'preco_por_m2', 'Quartos', 'Banheiros']])

median_rent = df_apartments['Aluguel Total'].median()

# Create a binary target variable ('Sim' if the rent is below or equal to the median, 'Não' if above)
df_apartments['is_good_rent'] = df_apartments['Aluguel Total'].apply(lambda x: 'Sim' if x <= median_rent else 'Não')


X = df_apartments.drop(columns=['Aluguel Total', 'is_good_rent', 'Link', 'Aluguel'])  # Features
y = df_apartments['is_good_rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

df_apartments_original['is_good_rent'] = df_apartments['is_good_rent']  # Adding is_good_rent to the original DataFrame

st.title("Aluguel de Apartamentos em Natal")

# Sidebar for filtering options
st.sidebar.header("Filtros")

# Use the saved 'Bairro' column for filtering
# Get unique neighborhoods
bairros = df_apartments_original['Bairro'].unique()

# Add "Select All" option to the neighborhoods list
bairros_with_select_all = ['Select All'] + list(bairros)

# Multiselect with "Select All" option
neighborhoods = st.sidebar.multiselect(
    "Selecione o(s) bairro(s)", 
    options=bairros_with_select_all, 
    default=bairros_with_select_all
)

# Logic to handle "Select All"
if "Select All" in neighborhoods:
    neighborhoods = list(bairros)  # Select all neighborhoods

    
min_price, max_price = st.sidebar.slider("Selecione uma faixa de preço", int(df_apartments_original['Aluguel'].min()), int(df_apartments_original['Aluguel'].max()), (1000, 3000))
good_price = st.sidebar.checkbox("Recomendados")
sort_by = st.sidebar.selectbox("Ordernar por:", options=['Bairro', 'Aluguel', 'Tamanho'], index=0)
sort_order = st.sidebar.radio("Ordem de ordenação", options=['Ascendente', 'Descendente'], index=0)

# Apply sorting

# Apply filters using original data
filtered_df = df_apartments_original[
    df_apartments_original['Bairro'].isin(neighborhoods) &
    (df_apartments_original['Aluguel'] >= min_price) &
    (df_apartments_original['Aluguel'] <= max_price)
]
filtered_df = filtered_df.rename(columns={'is_good_rent': 'Recomendado', 'Extras': 'Condominio (quando não incluso)'})
# Filter further if good prices checkbox is checked
if good_price:
    filtered_df = filtered_df[filtered_df['Recomendado'] == 'Sim']

if sort_order == 'Ascendente':
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=True)
else:
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)


# Create clickable links in the DataFrame
def make_clickable(link):
    return f'<a href="{link}" target="_blank">View Apartment</a>'

# Apply the clickable link to the Link column
filtered_df['Link'] = filtered_df['Link'].apply(make_clickable)

# Display the filtered DataFrame with original values in Streamlit
count_df = filtered_df.shape[0]  # Get the number of rows in the filtered DataFrame
st.markdown(f"### Apartamentos Filtrados: {count_df} ")
st.write(filtered_df.to_html(escape=False), unsafe_allow_html=True)


# Save the filtered data for download (optional)
st.sidebar.download_button(
    label="Download filtered data as CSV",
    data=filtered_df.to_csv().encode('utf-8'),
    file_name='filtered_apartments.csv',
    mime='text/csv',
)
