from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

x, y = make_regression(n_samples=200, n_features=5, noise=5, random_state=42)
xt, xe, yt, ye = train_test_split(x, y, test_size=.3, random_state=13)
model = TransformedTargetRegressor(
    regressor=make_pipeline(StandardScaler(), SVR(C=10, epsilon=.1, gamma='scale')),
    transformer=StandardScaler()).fit(xt, yt)
baseline = DummyRegressor(strategy='mean').fit(xt, yt)
for name, predictor in [('mean baseline', baseline), ('SVR', model)]:
    prediction = predictor.predict(xe)
    print(f'{name}: MAE={mean_absolute_error(ye,prediction):.3f} RMSE={root_mean_squared_error(ye,prediction):.3f}')
target_scale = model.transformer_.scale_[0]
print(f'training target SD={target_scale:.3f}; epsilon in original units={.1*target_scale:.3f}')
print('support rows:', len(model.regressor_.named_steps['svr'].support_))
