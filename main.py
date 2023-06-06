from datetime import timedelta
import yfinance as yf
import pandas as pd

from river import compose
from river import preprocessing
from river import metrics
from river import stream
from river import neural_net as nn
from river import optim


def prepare_data(ticker, start_date, end_date, interval, start_time='15:30', end_time='22:00'):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    all_data = pd.DataFrame()
    all_labels = pd.Series()

    while start_date < end_date:
        next_end_date = start_date + timedelta(days=7)
        if next_end_date > end_date:
            next_end_date = end_date

        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=start_date, end=next_end_date, interval=interval)

        data = data.tz_convert('Europe/Warsaw')
        data = data.between_time(start_time, end_time)
        data = data[['Close']].copy()

        data.fillna(method='ffill', inplace=True)
        
        label = data['Close'].shift(-1)[:-1]

        all_data = pd.concat([all_data, data[:-1]])
        all_labels = pd.concat([all_labels, label])

        start_date = next_end_date

    return all_data, all_labels


if __name__ == '__main__':
    X, Y = prepare_data("AAPL", "2023-05-01", "2023-06-06", "1m")

    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        nn.MLPRegressor(
            hidden_dims=(5,),
            activations=(
                nn.activations.ReLU,
                nn.activations.ReLU,
                nn.activations.Identity
            ),
            optimizer=optim.SGD(1e-3)
        )
    )

    metric = metrics.MAE()

    for x, y in stream.iter_pandas(X, Y):
        y_pred = model.predict_one(x)
        metric.update(y, y_pred)
        model.learn_one(x, y)

    print(f'accuracy: {metric}')
