import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_abline, ggsave

arquivo = open("X.txt", "r")
texto_X = []
while True:
    linha = arquivo.readline()
    if not linha:
        break
    texto_X.append(linha.rstrip("\n"))
arquivo.close()

arquivo = open("y.txt", "r")
texto_y = []
while True:
    linha = arquivo.readline()
    if not linha:
        break
    texto_y.append(linha.rstrip("\n"))
arquivo.close()

X = np.array(list(map(float, texto_X)))
y = np.array(list(map(float, texto_y)))

X_mat = np.column_stack((np.ones(len(X)), X))
y_mat = y.reshape(-1, 1)

beta = np.linalg.inv(X_mat.T @ X_mat) @ (X_mat.T @ y_mat)

a = beta[0][0]
b = beta[1][0]

df = pd.DataFrame({"x": X, "y": y})

plot = (
    ggplot(df, aes("x", "y"))
    + geom_point()
    + geom_abline(intercept=a, slope=b)
)

ggsave(plot, "grafico.png", dpi=300)

print(a, b)
