# Notebooks

このフォルダには、Kaggle「Titanic: Machine Learning from Disaster」で、乗客の生存・死亡予測に取り組んだNotebookを格納しています。

## Notebook

### [titanic_machine_learning.ipynb](titanic_machine_learning.ipynb)

Titanicの乗客データを用いて、生存者を予測する二値分類モデルを構築したNotebookです。

## Contents

Notebookでは、以下の順番で分析しています。

1. 学習データとテストデータの読み込み
2. データ型と欠損値の確認
3. 性別による生存率の比較
4. 客室クラスによる生存率の比較
5. 欠損値の補完
6. `FamilySize`の特徴量設計
7. カテゴリ変数のOne-Hot Encoding
8. Gradient Boostingモデルの構築
9. 5分割交差検証
10. 特徴量重要度の分析
11. Kaggle提出ファイルの作成

## Features

最終モデルでは、以下の特徴量を使用しています。

* `Pclass`：客室クラス
* `Sex`：性別
* `Age`：年齢
* `Fare`：運賃
* `Embarked`：乗船港
* `Parch`：同乗する親・子の人数
* `FamilySize`：本人を含む家族人数

`FamilySize`は、以下の計算で作成しています。

```python
FamilySize = SibSp + Parch + 1
```

## Model

最終モデルには、以下を使用しています。

```python
GradientBoostingClassifier(random_state=1)
```

前の決定木が間違えたデータを次の決定木で重点的に学習し、複数のモデルを順番に組み合わせる分類手法です。

## Results

| Metric                   |       Score |
| ------------------------ | ----------: |
| 5-Fold CV Accuracy       | **0.83054** |
| CV Standard Deviation    |     0.02069 |
| Best Kaggle Public Score | **0.78468** |

### Cross-Validation Details

|     Fold |   Accuracy |
| -------: | ---------: |
|        1 |     0.8156 |
|        2 |     0.8258 |
|        3 |     0.8483 |
|        4 |     0.8034 |
|        5 |     0.8596 |
| **Mean** | **0.8305** |

## Key Findings

* 女性の生存率は約74.2％、男性は約18.9％だった
* 客室クラスによって生存率に差が見られた
* 性別が最も重要度の高い特徴量だった
* 運賃、客室クラス、年齢、家族人数も予測に影響した
* 交差検証とKaggle Public Scoreには差が生じた

## Color Guide

Notebook内の説明は、目的に応じて色分けしています。

* 緑：最終予測に必要な処理
* 青：データやモデルを理解するための分析
* 黄：評価結果を解釈する際の注意点

## Notes

Kaggleのデータセットは、このリポジトリには含めていません。

Notebookを実行する場合は、KaggleのTitanicコンペティションに参加し、以下のファイルが利用できる環境で実行してください。

```text
train.csv
test.csv
```
