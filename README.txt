Esse trabalho é composto por 3 arquivos .py, uma pasta de dados e uma pasta de outputs.

-Para executar os algoritmos chame de seu terminal:

python3 main.py

	A função é responsável por chamar os algoritmos, rodar e salvar os resultados experimentais.
	Gera como saída outputs/resultados.csv onde as classificações obtidas para os dados de teste são salvas e centroids.txt onde as coordenadas das centroides gerada pelo k-means são armazenadas.

-Para obter as métricas e sumário de estatisticas e necessário rodar o comando:

python3 metricas.py

(alternativamente) python3 metricas.py > outputs/metricas_log.txt

	O script é responsável por printar na tela a matriz de confusão, acuracia e outras métricas para o algoritmo KNN e reporta as centroides e a contagem da classificação por grupo para cada label. Gera como saída tabelas no formato latex para as matrizes de confusão do KNN, métricas: precisão, revocação e F1 do KNN, contagens de labels por grupo no k-means e gráficos de grupos do k-means. Essas tabelas e gráficos são apresentadas em nossa documentação, junto com tabelas adicionais.