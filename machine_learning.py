from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pickle


class MachineLearning:
    def __init__(self):
        """Classe que encapsula funções de pré-processamento, modelagem e avaliação de Machine Learning."""
        pass

    @staticmethod
    def plot_histogram(data: pd.DataFrame, column: str, target_column: str, **kwargs) -> None:
        """
        Plota um histograma da coluna especificada.

        :param data: DataFrame contendo os dados.
        :param column: Nome da coluna a ser usada no eixo X.
        :param target_column: Nome da coluna usada para colorir os dados.
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        px.histogram(data, x=column, color=target_column, **kwargs).show()

    @staticmethod
    def plot_box(data: pd.DataFrame, column: str, target_column: str, **kwargs) -> None:
        """
        Plota um boxplot da coluna especificada.

        :param data: DataFrame contendo os dados.
        :param column: Nome da coluna a ser usada no eixo X.
        :param target_column: Nome da coluna usada para colorir os dados.
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        px.box(data, x=column, color=target_column, **kwargs).show()

    @staticmethod
    def get_target_column(data: pd.DataFrame, target: str) -> pd.Series:
        """
        Obtém a coluna alvo (target) do DataFrame.

        :param data: DataFrame contendo os dados.
        :param target: Nome da coluna alvo.
        :return: Série contendo os valores da coluna alvo.
        """
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")

        return data[target]

    @staticmethod
    def drop_column(data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove uma coluna do DataFrame.

        :param data: DataFrame contendo os dados.
        :param column: Nome da coluna a ser removida.
        :return: DataFrame sem a coluna especificada.
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        return data.drop(column, axis=1)

    @staticmethod
    def get_one_hot(columns: list) -> ColumnTransformer:
        """
        Cria um transformador para aplicar One-Hot Encoding nas colunas especificadas.

        :param columns: Lista de colunas categóricas para transformação.
        :return: Objeto ColumnTransformer configurado para One-Hot Encoding.
        """
        one_hot = make_column_transformer(
            (OneHotEncoder(drop='if_binary'), columns),
            remainder='passthrough',
            sparse_threshold=0
        )
        return one_hot

    @staticmethod
    def one_hot_transform_data(one_hot: ColumnTransformer, data: pd.DataFrame):
        """
        Aplica o One-Hot Encoding nos dados.

        :param one_hot: Objeto ColumnTransformer configurado.
        :param data: DataFrame contendo os dados originais.
        :return: Dados transformados.
        """
        return one_hot.fit_transform(data)

    @staticmethod
    def one_hot_transform(one_hot: ColumnTransformer, data: pd.DataFrame):
        """
        Transforma um novo conjunto de dados com One-Hot Encoding já treinado.

        :param one_hot: Objeto ColumnTransformer treinado.
        :param data: Novo DataFrame a ser transformado.
        :return: Dados transformados.
        """
        return one_hot.transform(data)

    @staticmethod
    def dummy_column(data: pd.Series):
        """
        Converte variáveis categóricas em numéricas usando Label Encoding.

        :param data: Série categórica a ser transformada.
        :return: Série transformada em valores numéricos.
        """
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(data)

    @staticmethod
    def split_test_data(data: pd.DataFrame, target: pd.DataFrame, **kwargs):
        """
        Divide os dados em treino e teste.

        :param data: DataFrame de entrada.
        :param target: Coluna alvo.
        :return: Conjunto de treino e teste para features e target.
        """
        return train_test_split(data, target, **kwargs)

    @staticmethod
    def get_dummy_fit(data: pd.DataFrame, target: pd.DataFrame):
        """
        Treina um modelo Dummy para comparação de performance.

        :param data: Dados de treino.
        :param target: Target de treino.
        :return: Modelo Dummy treinado.
        """
        dummy = DummyClassifier()
        return dummy.fit(data, target)

    @staticmethod
    def get_dummy_score(dummy: DummyClassifier, data: pd.DataFrame, target: pd.DataFrame):
        """
        Obtém o score do modelo Dummy.

        :param dummy: Modelo Dummy treinado.
        :param data: Dados de teste.
        :param target: Target de teste.
        :return: Score do modelo.
        """
        return dummy.score(data, target)

    @staticmethod
    def get_tree_fit(data: pd.DataFrame, target: pd.DataFrame, **kwargs):
        """
        Treina um modelo de Árvore de Decisão.

        :param data: Dados de treino.
        :param target: Target de treino.
        :return: Modelo treinado.
        """
        tree = DecisionTreeClassifier(random_state=5, **kwargs)
        return tree.fit(data, target)

    @staticmethod
    def get_tree_predict(tree: DecisionTreeClassifier, data: pd.DataFrame):
        """
        Realiza predições com o modelo de Árvore de Decisão.

        :param tree: Modelo treinado.
        :param data: Dados de teste.
        :return: Predições do modelo.
        """
        return tree.predict(data)

    @staticmethod
    def get_tree_score(tree: DecisionTreeClassifier, data: pd.DataFrame, target: pd.DataFrame):
        """
        Obtém o score do modelo de Árvore de Decisão.

        :param tree: Modelo treinado.
        :param data: Dados de teste.
        :param target: Target de teste.
        :return: Score do modelo.
        """
        return tree.score(data, target)

    @staticmethod
    def plot_results(tree: DecisionTreeClassifier, class_names: list, **kwargs):
        """
        Plota a árvore de decisão.

        :param tree: Modelo treinado.
        :param class_names: Nomes das classes.
        """
        plt.figure(figsize=(15, 7))
        plot_tree(tree, class_names=class_names, **kwargs)
        plt.show()

    @staticmethod
    def get_min_max():
        """
        Retorna um objeto MinMaxScaler para normalização.

        :return: Objeto MinMaxScaler.
        """
        return MinMaxScaler()

    @staticmethod
    def minmax_fit_transform(minmax: MinMaxScaler, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza os dados usando MinMaxScaler.

        :param minmax: Objeto MinMaxScaler.
        :param data: Dados a serem normalizados.
        :return: Dados normalizados.
        """
        normalized_data = minmax.fit_transform(data)
        return pd.DataFrame(normalized_data)

    @staticmethod
    def get_knn_fit(data: pd.DataFrame, target: pd.DataFrame) -> KNeighborsClassifier:
        """
        Treina um modelo KNN.

        :param data: Dados de treino.
        :param target: Target de treino.
        :return: Modelo KNN treinado.
        """
        knn = KNeighborsClassifier()
        return knn.fit(data, target)

    @staticmethod
    def get_knn_score(knn: KNeighborsClassifier, data: pd.DataFrame, target: pd.DataFrame) -> float:
        """
        Obtém o score do modelo KNN.

        :param knn: Modelo treinado.
        :param data: Dados de teste.
        :param target: Target de teste.
        :return: Score do modelo.
        """
        return knn.score(data, target)

    @staticmethod
    def pickle_dump(dump, name: str):
        """
        Salva um modelo em um arquivo pickle.

        :param dump: Objeto a ser salvo.
        :param name: Nome do arquivo.
        """
        with open(f"model_{name}.pkl", 'wb') as file:
            pickle.dump(dump, file)
