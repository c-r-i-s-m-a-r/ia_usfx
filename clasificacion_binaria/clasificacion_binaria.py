import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from matplotlib import pyplot
    import numpy as np
    import pandas as pd
    import crismar_tools as cst

    return cst, np, pd, pyplot


@app.cell
def _(np):
    def sigmoide(z):
        z = np.array(z)
        return 1 / (1 + np.exp(-z))

    return (sigmoide,)


@app.cell
def _(np, sigmoide, y):
    def costo(_x, _y, _theta):
        m = y.size
        h = sigmoide(_x.dot(_theta.T))
        cost = np.sum(-_y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h)) ) / m
        return cost

    return (costo,)


@app.cell
def _(costo, sigmoide):
    def descenso_gradiente(_x, _y, _theta, _alpha, num_iters):
        _m = _y.size
        _theta = _theta.copy()
        _history = []
        for i in range(num_iters):
            _h = sigmoide( _x.dot(_theta) )
            _theta = _theta - (_alpha / _m) * (_h - _y).dot(_x)
            _history.append(costo(_x, _y, _theta))
        return _theta, _history

    return (descenso_gradiente,)


@app.cell
def _(pd):
    df_train = pd.read_csv('clasificacion/train.csv')
    return (df_train,)


@app.cell
def _(df_train):
    df_train.info(verbose=True)
    return


@app.cell
def _(df_train):
    df_train.shape
    return


@app.cell
def _(df_train):
    index_y = df_train.shape[1] - 1
    x_noprc = df_train.iloc[:, :index_y]
    y = df_train.iloc[:, index_y]
    return x_noprc, y


@app.cell
def _(cst, x_noprc):
    x, mu, sigma = cst.preprocesamiento.normalizar_oneHot(x_noprc)
    return (x,)


@app.cell
def _(x):
    x.insert(0, 'x0', 1)
    return


@app.cell
def _(x):
    x.shape
    return


@app.cell
def _(y):
    y.shape
    return


@app.cell
def _(np, x):
    theta = np.zeros(x.shape[1])
    theta.shape
    return (theta,)


@app.cell
def _(descenso_gradiente, theta, x, y):
    n_iter = 10000
    alpha = 0.003
    theta_f, history = descenso_gradiente(x, y, theta, alpha, n_iter)
    return (history,)


@app.cell
def _(history, np, pyplot):
    pyplot.plot(np.arange(len(history)), history)
    pyplot.xlabel('numero de iteraciones')
    pyplot.ylabel('costo')
    pyplot.show()
    return


@app.cell
def _(pd):
    df_test = pd.read_csv('clasificacion/test.csv')
    return (df_test,)


@app.cell
def _(df_test):
    index_y_test = df_test.shape[1] - 1
    x_test = df_test.iloc[:, :index_y_test]
    y_test = df_test.iloc[:, index_y_test]
    return


app._unparsable_cell(
    r"""
    x_test_norm = cst.normalizar_oneHot(x_test, mu, sigma, True)
    x_test_norm.insert(0, 'x0', 1)
    readmisiones = sigmoide(x_test_norm.dot(theta_f))
    #pd.DataFrame(readmisiones, columns=['Readmitido'])
    2readmisiones
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
