import pandas as pd
from machine_learning import MachineLearning as ML  # Importa a classe MachineLearning para utilizar métodos de processamento e modelagem


class Main:
    def __init__(self):
        """
        Classe principal responsável por carregar os dados e executar o pipeline de Machine Learning.
        """
        pd.set_option('display.max_columns', 200)  # Configura o pandas para exibir até 200 colunas
        self.data = pd.read_csv('churn.csv')  # Carrega os dados a partir de um arquivo CSV

    def process(self):
        """
        Executa o pipeline de processamento e modelagem dos dados.
        """
        # Obtém a coluna-alvo 'churn'
        target = ML.get_target_column(self.data, "churn")

        # Remove colunas irrelevantes para o modelo ('id_cliente' e 'churn')
        data = ML.drop_column(self.data, "id_cliente")
        data = ML.drop_column(data, "churn")

        # Aplica One-Hot Encoding às colunas categóricas
        one_hot = ML.get_one_hot(columns=["pais", "sexo_biologico"])
        data = ML.one_hot_transform_data(one_hot=one_hot, data=data)

        # Converte a coluna-alvo para valores numéricos
        target = ML.dummy_column(target)

        # Divide os dados em conjunto de treino e teste, garantindo estratificação para manter a proporção da classe-alvo
        data_train, data_test, target_train, target_test = ML.split_test_data(data, target, stratify=target, random_state=5)

        # Modelo Dummy (modelo base para comparação)
        dummy = ML.get_dummy_fit(data_train, target_train)
        score = ML.get_dummy_score(dummy, data_test, target_test)
        print(f"The score of the dummy model is {score}")  # Exibe o desempenho do modelo dummy

        # Modelo de Árvore de Decisão com profundidade máxima de 4
        tree = ML.get_tree_fit(data_train, target_train, max_depth=4)
        ML.get_tree_predict(tree, data_test)  # Realiza a predição nos dados de teste
        score = ML.get_tree_score(tree, data_test, target_test)
        print(f"The score of the tree model is {score}")  # Exibe a acurácia do modelo de árvore

        # Plota a árvore de decisão
        ML.plot_results(tree, class_names=['não', 'sim'], fontsize=5, filled=True)

        # Normalização dos dados para uso no modelo KNN
        minmax = ML.get_min_max()
        normalized_data_train = ML.minmax_fit_transform(minmax, data_train)
        knn = ML.get_knn_fit(normalized_data_train, target_train)
        normalized_data_test = ML.minmax_fit_transform(minmax, data_test)
        score = ML.get_knn_score(knn, normalized_data_test, target_test)
        print(f"The KNN model score is {score}")  # Exibe o desempenho do modelo KNN

        # Salva os modelos processados usando pickle
        ML.pickle_dump(one_hot, "one_hot")
        ML.pickle_dump(tree, "tree")

        # Carrega os modelos salvos
        one_hot_pickle = pd.read_pickle("model_one_hot.pkl")
        tree_model_pickle = pd.read_pickle("model_tree.pkl")

        # Novo conjunto de dados para predição
        new_data = pd.DataFrame({
            'score_credito': [850],
            'pais': ['França'],
            'sexo_biologico': ['Homem'],
            'idade': [27],
            'anos_de_cliente': [3],
            'saldo': [56000],
            'servicos_adquiridos': [1],
            'tem_cartao_credito': [1],
            'membro_ativo': [1],
            'salario_estimado': [85270.00]
        })

        # Transforma os dados do novo cliente usando o One-Hot Encoding salvo
        normalized_new_data = ML.one_hot_transform(one_hot_pickle, new_data)

        # Realiza a predição com o modelo salvo
        result = tree_model_pickle.predict(normalized_new_data)

        print(f"The prediction for the pickle tree model on the new data was {result}")  # Exibe o resultado da predição


if __name__ == '__main__':
    main = Main()
    main.process()  # Executa o pipeline de Machine Learning
