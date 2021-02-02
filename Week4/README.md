# Chapter.5.1 - Regression
## 1. Introduction

- 정의 : 통계학적으로 회귀는 여러 개의 독립변수와 한 개의 종속변수 간의 상관관계를 모델링한 기법을 통칭

- 유래 : 영국의 통계학자 갈톤(Galton)이 수행한 연구에서 유래

    부모와 자식 간의 키의 상관관계를 분석한 결과, 

    - 키가 큰 부모의 자식이 언제나 부모보다 크지않고, 키가 작은 부모의 자식이 언제나 부모보다 작지 않다
    - 부모의 키가 아주 크더라도 자식의 키가 부모보다 더 커서 무한정 커지지는 않는다
    - 부모의 키다 아주 작더라도 자식의 키가 부모보다 더 작아서 무한정 작아지지는 않는다

    **즉, 사람의 키는 평균 키로 회귀하려는 경향을 가진다는 자연의 법칙**

- 선형 회귀식

    $$Y = W_1 * X_1 + W_2 * X_2 + W_3 * X_3 ... + W_n * X_n $$

    (W는 독립변수의 값에 영향을 미치는 회귀 계수, X는 독립변수, Y는 종속변수)

😎머신러닝 회귀 예측의 핵심은 주어진 피처와 결정 값 데이터 기반에서 학습을 통해 **최적의 회귀 계수**를 찾아내는 것

- 머신러닝의 지도학습(Supervised Learning)인 분류와 회귀의 결정적인 차이점은 예측 결과값이 이산형 데이터인지, 연속형 데이터인지의 차이

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5bcb4d4f-8fcd-4c3e-a246-260300816c77/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5bcb4d4f-8fcd-4c3e-a246-260300816c77/Untitled.png)

- 형태에 따른 회귀 방식 구분
    - 독립변수의 개수 : 1개 - 단일 회귀 / 여러개 - 다중 회귀
    - 회귀 계수의 결합 : 선형 - 선형회귀 / 비선형 - 비선형회귀

- 규제 방법(Regularization)에 따른 선형 회귀 유형 구분

    (※ 선형 회귀가 가장 많이 쓰이므로 우선 다루기로 한다)

    (※ 규제 : 과적합 문제를 해결하기 위해 회귀 계수에 페널티 값을 적용하는 것을 말함)

    - 규제 방법에 따른 선형 회귀 유형

# 2. Simple Linear Regression Model

- 정의 : 독립 변수와 종속변수가 하나인 선형 회귀
- Example) 주택크기와(독립변수) 주택가격간의 선형 관계를 나타내는 단순 선형 회귀 모델
    - X축이 주택의 크기, Y축이 주택 가격 축인 2차원 평면에 대한 예측을 위한 회귀식은 다음과 같다

    $$Y = W_0 + W_1 * X$$

    여기서 실제 주택 가격은 예측 함수 값에서 실제 값만큼 뺀(또는 더한)값이며 이 차이를 잔차라고(Error) 부른다

    이를 식에 적용해 보면 다음과 같다

    $$Y = W_0 + W_1 * X + Error$$

    최적의 회귀 모델을 만든다는 것은 바로 전체 데이터의 잔차의 합을 최소화시키는 모델을 만든다는 의미

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/09afff21-b5ec-4036-962d-9bc29be11a34/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/09afff21-b5ec-4036-962d-9bc29be11a34/Untitled.png)

- 오류 값은 +또는 -가 될 수 있지만, 전체 데이터의 오류 합을 구하기 위해 단순히 더하면 오류의 합이 크게 줄 수 있음 (*Residual = +1 -1 = 0)
- 따라서 오류 합을 계산할 때는 절대값을 취해서 더하거나 (MAE : Mean Absolute Error), 오류 값의 제곱을 구해서 더하는 방식(RSS : Residual Sum of Square)를 취함

    (미분 등의 계산을 편리하기 위해서 RSS가 일반적으로 사용됨.)

    $$Error^2 = RSS$$

    - Example) 주택 가격 예제에서의 RSS

        $$RSS = (HousePrice_1 - (W_0 + W_1 * HouseSize_1))^2 +(HousePrice_2 - (W_0 + W_1 * HouseSize_2))^2
        ... (모든 학습데이터에 대해 RSS 수행)$$

- RSS는 회귀식의 독립변수 X, 종속변수 Y가 중심변수가 아니라, 회귀 계수 W가 중심임을 인지해야 함
- 일반적인 RSS의 정규화 식

    $$RSS(w_0, w_1) = \frac1N \Sigma (y_i -(w_0 + w_1 * x_i)^2 $$

- 회귀에서 RSS는 비용(Cost)이며, W로 구성되는 RSS를 비용 함수(Cost Function or Loss Function)이라 부름
- 머신러닝의 회귀 알고리즘의 목적은 데이터를 계속 학습하여 이 비용 함수가 반환하는 값을 지속해서 감소시켜서 최종적으로는 더 이상 감소하지 않는 최소의 오류 값을 구하는 것을 말함

# 3. Gradient Descent

- 정의 : 비용함수 RSS를 최소화하기 위해서 '점직적으로' 반복적인 계산을 통해 W파라미터 값을 업데이트하면서 오류 값이 최소가 되는 W 파라미터를 구하는 방식
- 핵심은 "어떻게 하면 오류가 작아지는 방향으로 W값을 보정할 수 있을까?"
- 예를 들어 비용 함수가 2차 함수라면, 최초 W에서부터 미분을 적용한 뒤 이 미분 값이 계속 감소하는 방향으로 순차적으로 W를 업데이트함
- 미분된 1차 함수의 기울기가 감소하지 않는 지점을 비용 함수의 최소 지점으로 간주 그때의 W값을 반환
- 경사하강법 방식
    - Step 1: 임의의 두 지점 w1, w0에서 첫 비용 함수의 값을 계산

        $$RSS(W) = \frac1N \Sigma (y_i -(w_0 + w_1 * x_i)^2 $$

    - Step 2: R(W)는 두 개의 w 파라미터인 w0와 w1를 각각 가지고 있기에 각 변수를 순차적으로 편미분을 수행하여 R(W)를 최소화하는 w0와 w1를 얻어야 함

    $$\frac {\partial R(w)}{\partial w_1} = \frac2N \Sigma^N_{i=1} -x_t(y_i -(w_0 + w_1 * x_i) =  -\frac2N \Sigma^N_{i=1} x_t(실제값_i - 예측값)$$

    $$\frac {\partial R(w)}{\partial w_0} = \frac2N \Sigma^N_{i=1} -(y_i -(w_0 + w_1 * x_i) =  -\frac2N \Sigma^N_{i=1}(실제값_i - 예측값)$$

    - 이렇게 구한 편미분 결과값으로 w1과 w0를 반복적으로 보정하면서 업데이트
    - 단, 편미분 값이 너무 클 수 있기 때문에 보정 계수(Learning Rate) η를 비용 함수에 곱하여 w0와 w1값을 서서히 업데이트 (여기서 편미분 결괏값을 빼줘야하기 때문에 +로 바뀐것임)

        $$w_1 = w_1+\eta\frac2N \Sigma^N_{i=1} x_t(실제값_i - 예측값)$$

        $$w_0 = w_0 + \eta\frac2N \Sigma^N_{i=1}(실제값_i - 예측값)$$

    - 비용 함수의 값이 감소했으면 다시 Step 2를 반복하여 더 이상 비용 함수의 값이 감소하지 않으면 그때의 W1, W0를 구하고 반복을 중지

## 4. LinearRegression 클래스 - Ordinary Least Squares

- LinearRegression : 사이킷런에서 규제가 적용되지 않은 선형 회귀를 구현한 클래스
- 특징

    ① 예측값과 실제 값의 RSS(Residual Sum of Squares)를 최소화해 OLS(Ordinary Least Squares) 추정 방식으로 구현함

    ② Fit() 메서드로 x, y 배열을 입력 받으면 회귀 계수 (Coefficient)인 w를 coef_ 속성에 저장함

    ③ 입력 파라미터 종류

    fit_intercept : 절편(intercept) 값을 계산할 것인지 말지를 지정함. Default값은 True. (False는 계산하지 않음)

    normalize : 데이터 세트를 정규화할지 말지를 지정함. Default값은 False. (True는 정규화함)

    ④ 속성

    coef_ : fit() 메서드를 수행했을 때 회귀 계수가 배열 형태로 저장하는 속성

    intercept_ : 절편값

    ⑤ OLS 기반의 회귀 계수 계산은 입력 피처의 독립성에 많은 영향을 받음. 즉, 피처간의 상관관계가 높은 경우 분산이 매우 커져서 오류에 매우 민감해짐 (다중공선성(Multi-collinearity)문제)

    다중 공선성 문제가 발생한다면, PCA를 통해 차원 축소를 수행하는 것을 고려해야함

## 5. 회귀 평가 지표

- 회귀의 평가 지표는 실제 값과 회귀 예측값의 차이 값을 기반으로 한 지표가 중심임

    다만, 이 차이를 그냥 모두 더해버리면 +와 -가 섞여서 오류가 상쇄될 수도 있기에 절댓값 평균이나 제곱, 또는 제곱한 뒤 루트를 씌운 평균으로 평가함

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b83e42ba-6eec-494a-973a-a5ebb510dfc9/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b83e42ba-6eec-494a-973a-a5ebb510dfc9/Untitled.png)

- Cross_val_score, GridSearchCV와 같은 Scoring 함수에 회귀 평가 지표를 적용할 때 유의할 점

    예를 들어, MAE의 Scoring 파라미터에 'neg_mean_absolute_error'와 같이 'neg_'라는 접두어가 붙어있는 경우 이는 Negative(음수)값을 가진다는 의미임

- 음수를 적용해야 하는 이유는 Scoring 함수의 경우 값이 클수록 좋은 평과 결과라고 판단하기 때문임

## 6. 다항 회귀 이해

- 정의 : 회귀가 독립변수의 단항식이 아닌 2차, 3차 방정식과 같은 다항식으로 표현되는 것을 다항(Polynomial) 회귀라고 함

- 중요 : 다항 회귀 또한 선형 회귀임.

    (선형/비선형 회귀를 나누는 기준은 회귀 계수의 형태에 따른 것임)

- 사이킷런에서는 다항 회귀를 위한 클래스를 명시적으로 제공하지는 않음
    - 다항 회귀 역시 선형 회귀이기 때문에 비선형 함수를 선형 모델에 적용시키는 방법을 사용
    - PolynomialFeature 클래스의 degree 파라미터를 통해 입력 받은 단항식 피처를 degree에 해당하는 다항식 피처로 변환함 [ fit(), transform() 메서드를 통해 변환작업 수행 ]

        ex) Poly = PolynomialFeatures(degree = 2).fit_transform(x)

## 7. 다항 회귀를 이용한 과소적합 및 과적합 이해

- 다항 회귀는 다항식의 차수(degree)가 높아 질수록 학습 데이터에만 너무 맞춘 학습이 이뤄져서 테스트 환경에서는 오히려 예측 정확도가 떨어짐. (=과적합 문제)
- 좋은 예측 모델은 학습 데이터의 패턴을 지나치게 단순화한 과소적합 모델도 아니고, 지나치게 복잡한 과적합 모델도 아닌, 학습 데이터의 패턴을 잘 반영하면서도 복잡하지 않은 균형 잡힌 모델임
- 사이킷런 홈페이지의 예시 참조바람

    [Underfitting vs. Overfitting - scikit-learn 0.24.1 documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py)
