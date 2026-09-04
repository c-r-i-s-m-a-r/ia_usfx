import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from matplotlib import pyplot
    import pandas
    import numpy

    return OneHotEncoder, OrdinalEncoder, numpy, pandas, pyplot


@app.cell
def _(OrdinalEncoder):
    def encoder_ordinal(_df, columna, orden):
        df_enc = _df.copy()
        encoder = OrdinalEncoder(
            categories=[orden],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )

        df_enc[columna] = encoder.fit_transform(df_enc[[columna]])
        return df_enc, encoder

    return (encoder_ordinal,)


@app.cell
def _(OneHotEncoder, pandas):
    def encoder_oneHot(_df, columnas):
        encoder = OneHotEncoder(sparse_output=False)

        array_enc = encoder.fit_transform(_df[columnas])
        df_enc = _df.copy()

        df_tmp = pandas.DataFrame(
            array_enc,
            columns=encoder.get_feature_names_out(columnas),
            index=df_enc.index
        )

        df_enc = pandas.concat(
            [df_enc.drop(columns=columnas), df_tmp], 
            axis=1
        )
        return df_enc, encoder


    return (encoder_oneHot,)


@app.cell
def _(pandas):
    def encoder_oneHot_infer(_df, _columnas, _encoder):
        array_enc = _encoder.transform(_df[_columnas])

        columnas_train = _encoder.get_feature_names_out(_columnas)

        df_tmp = pandas.DataFrame(
            array_enc,
            columns=columnas_train,
            index=_df.index
        )

        return pandas.concat([_df.drop(columns=_columnas), df_tmp], axis=1)

    return (encoder_oneHot_infer,)


@app.cell
def _(numpy):
    def normalizar(_df):
        df_norm = _df.copy()
        n = df_norm.shape[1]

        media = numpy.zeros(n)
        des_std = numpy.zeros(n)

        media = numpy.mean(_df, axis=0)
        des_std = numpy.std(_df, axis=0)

        df_norm = (df_norm - media) / des_std
        return df_norm, media, des_std

    return


@app.cell
def _(numpy, pandas):
    def normalizar_oneHot(_df, _mu=0, _sigma=0, bMuSigma=False):
        df_norm = _df.copy()
        media = []
        des_std = []

        if not bMuSigma:
            for columna in df_norm.columns:
                if max(df_norm[columna]) == 1 and min(df_norm[columna]) == 0:
                    media.append(0)
                    des_std.append(1)
                    continue
                _media = numpy.mean(df_norm[columna])
                _des_std = numpy.std(df_norm[columna])
                df_norm[columna] = (df_norm[columna] - _media) / _des_std
                media.append(_media)
                des_std.append(_des_std)
        else:
            for columna in df_norm.columns:
                if max(df_norm[columna]) == 1 and min(df_norm[columna]) == 0:
                    continue
                df_norm[columna] = (df_norm[columna] - _mu[columna].values[0]) / _sigma[columna].values[0]
            return df_norm

        media = pandas.DataFrame([media], columns=df_norm.columns)
        des_std = pandas.DataFrame([des_std], columns=df_norm.columns)
        return df_norm, media, des_std

    return (normalizar_oneHot,)


@app.cell
def _(numpy):
    def costo(_x, _y, _theta):
        m = _x.shape[0]
        h = numpy.dot(_x, _theta)
        J = numpy.sum( numpy.square( h - _y ) ) / (2 * m)
        return J

    return (costo,)


@app.cell
def _(costo, numpy):
    def descenso_gradiente(_x, _y, _theta, _alpha, _n_iter):
        _m = _x.shape[0]
        theta_c = _theta.copy()
        history = []
        for i in range(_n_iter):
            h = numpy.dot(_x, theta_c)
            theta_c = theta_c - _alpha / _m * numpy.dot(h - _y, _x)
            history.append( costo(_x, _y, theta_c) )
        return theta_c, history

    return (descenso_gradiente,)


@app.cell
def _(pandas):
    dataset = pandas.read_csv('salary_prediction_data.csv')
    return (dataset,)


@app.cell
def _(dataset):
    dataset.shape
    return


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(dataset):
    index_y = dataset.shape[1] - 1
    X_no_enc = dataset.iloc[:, :index_y]
    Y = dataset.iloc[:, index_y]
    return X_no_enc, Y


@app.cell
def _(X_no_enc, encoder_oneHot, encoder_ordinal):
    orden_estudio = ['High School', 'Bachelor', 'Master', 'PhD']
    X, enc_estudio = encoder_ordinal(X_no_enc, 'Education', orden_estudio)
    X, enc_oneHot = encoder_oneHot(X, ['Location', 'Job_Title', 'Gender'])
    return X, enc_oneHot, orden_estudio


@app.cell
def _(X):
    X
    return


@app.cell
def _(X, normalizar_oneHot):
    X_norm, mu, sigma = normalizar_oneHot(X)
    X_norm
    return X_norm, mu, sigma


@app.cell
def _(X_norm, Y, descenso_gradiente, numpy):
    m = X_norm.shape[1]
    n_iter = 10000
    alpha = 0.001
    theta = numpy.zeros(m)
    theta, history = descenso_gradiente(X_norm, Y, theta, alpha, n_iter)
    theta
    return history, n_iter, theta


@app.cell
def _(history, n_iter, numpy, pyplot):
    pyplot.plot(numpy.arange(n_iter), history)
    pyplot.xlabel('Numero de iteraciones')
    pyplot.ylabel('Costo')
    return


@app.cell
def _(X_no_enc):
    persona1 = X_no_enc.iloc[0:1, :]
    persona1['Education'] = 'PhD'
    persona1['Experience'] = 18
    persona1['Location'] = 'Suburban'
    persona1['Job_Title'] = 'Director'
    persona1['Age'] = 38
    persona1['Gender'] = 'Male'
    persona1
    return (persona1,)


@app.cell
def _(
    enc_oneHot,
    encoder_oneHot_infer,
    encoder_ordinal,
    mu,
    normalizar_oneHot,
    orden_estudio,
    persona1,
    sigma,
):
    p1, null = encoder_ordinal(persona1, 'Education', orden_estudio)
    p1 = encoder_oneHot_infer(p1, ['Location', 'Job_Title', 'Gender'], enc_oneHot)
    p1 = normalizar_oneHot(p1, mu, sigma, True)
    p1
    return (p1,)


@app.cell
def _(p1, theta):
    salario = p1.dot(theta)
    salario
    return


if __name__ == "__main__":
    app.run()
