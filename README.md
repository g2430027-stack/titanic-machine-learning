# Titanic Survival Prediction

Kaggleの「Titanic: Machine Learning from Disaster」を題材に、乗客の属性から生存・死亡を予測する二値分類モデルを構築しました。

データの確認、欠損値処理、カテゴリ変数の変換、特徴量設計、Gradient Boostingによるモデル構築、5分割交差検証、特徴量重要度の分析まで行っています。

## Results

| Metric                   |       Score |
| ------------------------ | ----------: |
| 5-Fold CV Accuracy       | **0.83054** |
| CV Standard Deviation    |     0.02069 |
| Best Kaggle Public Score | **0.78468** |

交差検証では約83.1％の精度を記録しました。一方、KaggleのPublic Scoreとは差が生じたため、検証データと未知データの分布差や、モデルの汎化性能について考察しました。

## Notebook

分析・モデル構築・評価の詳細は、以下のNotebookにまとめています。

[Titanic Machine Learning Notebook](notebooks/titanic_machine_learning.ipynb)

## Analysis Flow

1. 学習データとテストデータの読み込み
2. データ型と欠損値の確認
3. 性別による生存率の比較
4. 客室クラスによる生存率の比較
5. 欠損値の補完
6. `FamilySize`の作成
7. カテゴリ変数のOne-Hot Encoding
8. Gradient Boostingモデルの構築
9. 5分割交差検証
10. 特徴量重要度の分析
11. Kaggle提出ファイルの作成

## Exploratory Data Analysis

### Survival Rate by Sex

| Sex    | Survival Rate |
| ------ | ------------: |
| Female |        0.7420 |
| Male   |        0.1889 |

女性と男性で生存率に大きな差が見られたため、`Sex`は重要な特徴量になると考えました。

### Passenger Class

客室クラス`Pclass`ごとの生存率も比較しました。上位クラスほど生存率が高い傾向があり、乗船位置や社会的背景に関係する重要な特徴量として採用しました。

## Preprocessing

### Missing Values

モデルで使用する欠損値は、以下の方法で補完しました。

| Feature    | Method    |
| ---------- | --------- |
| `Age`      | 学習データの中央値 |
| `Fare`     | 学習データの中央値 |
| `Embarked` | 学習データの最頻値 |

テストデータの欠損値にも学習データから計算した値を使用し、未知データの情報が学習処理に入らないようにしました。

### Categorical Variables

`Sex`と`Embarked`にはOne-Hot Encodingを適用しました。

```python
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

X_test = X_test.reindex(
    columns=X.columns,
    fill_value=0
)
```

学習データとテストデータの列を揃えることで、カテゴリの種類が異なる場合にも対応しています。

## Feature Engineering

### FamilySize

同乗している兄弟・配偶者の人数`SibSp`と、親・子の人数`Parch`から、本人を含む家族人数を作成しました。

```python
FamilySize = SibSp + Parch + 1
```

単独乗船か家族での乗船かによって、避難時の行動や生存確率が異なる可能性があると考えて追加しました。

## Features

最終モデルでは、以下の特徴量を使用しました。

* `Pclass`：客室クラス
* `Sex`：性別
* `Age`：年齢
* `Fare`：運賃
* `Embarked`：乗船港
* `Parch`：同乗する親・子の人数
* `FamilySize`：本人を含む家族人数

`PassengerId`、`Name`、`Ticket`、`Cabin`などは、今回の最終モデルでは使用していません。

## Model

最終モデルには`GradientBoostingClassifier`を使用しました。

```python
model = GradientBoostingClassifier(
    random_state=1
)
```

Gradient Boostingは、前の決定木が間違えたデータを次の決定木で重点的に学習し、複数のモデルを順番に組み合わせて予測性能を高める手法です。

学習過程ではRandom Forestも試しましたが、最終NotebookではGradient Boostingを採用しています。

## Cross-Validation

モデル評価には5分割交差検証を使用しました。

|     Fold |   Accuracy |
| -------: | ---------: |
|        1 |     0.8156 |
|        2 |     0.8258 |
|        3 |     0.8483 |
|        4 |     0.8034 |
|        5 |     0.8596 |
| **Mean** | **0.8305** |

1回のデータ分割だけで評価せず、異なる5つの分割で精度を確認しました。標準偏差は約0.0207で、分割によって約2ポイント程度のばらつきがありました。

## Feature Importance

最終モデルの特徴量重要度では、主に以下の特徴量が予測へ影響していました。

| Feature      | Importance |
| ------------ | ---------: |
| `Sex_female` |     0.3026 |
| `Sex_male`   |     0.1706 |
| `Fare`       |     0.1620 |
| `Pclass`     |     0.1469 |
| `Age`        |     0.1225 |
| `FamilySize` |     0.0750 |

性別の重要度が最も高く、探索的データ分析で確認した生存率の大きな差とも一致しました。

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* Matplotlib
* Jupyter Notebook
* Kaggle

## Repository Structure

```text
titanic-machine-learning/
├── README.md
├── notebooks/
│   ├── README.md
│   └── titanic_machine_learning.ipynb
├── .gitignore
└── LICENSE
```

Kaggleのデータセットはリポジトリに含めていません。

## What I Learned

この取り組みを通じて、欠損値処理、カテゴリ変数の変換、特徴量設計、分類モデルの構築、交差検証、特徴量重要度の確認という、表形式データ分析の基本的な流れを学びました。

また、交差検証の精度が向上してもKaggleのスコアが同じように向上するとは限らないことを経験しました。そのため、1つの評価値だけで判断せず、分割ごとのばらつきや未知データへの汎化性能を確認することが重要だと学びました。

## Future Improvements

* 前処理をPipeline化して各CV fold内で補完値を計算する
* `StratifiedKFold`で生存・死亡の比率を維持する
* 年齢を客室クラスや敬称ごとに補完する
* 敬称やチケットを使った特徴量を再検証する
* 家族単位・チケット単位の特徴量を作成する
* Logistic Regressionなどの単純なモデルと比較する
* Precision、Recall、F1-score、混同行列も確認する

## License

This project is licensed under the MIT License.

