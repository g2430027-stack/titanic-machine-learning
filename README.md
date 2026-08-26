# Titanic Survival Prediction

Kaggleの「Titanic: Machine Learning from Disaster」を題材に、
乗客の属性から生存者を予測する機械学習モデルを構築しました。

データの確認、欠損値処理、カテゴリ変数の変換、特徴量設計、
交差検証、ハイパーパラメータ調整を段階的に行いました。

## Result

- Best Cross-Validation Accuracy: **0.8305**
- Kaggle Public Score: **0.78468**

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Random Forest
- Kaggle

## Main Improvements

- `Age`の欠損値補完
- `Sex`や`Embarked`などのカテゴリ変数の変換
- `FamilySize`の作成
- `IsAlone`や`Title`などの特徴量を検証
- 交差検証によるモデル評価
- Random Forestのハイパーパラメータ調整
- CVスコアとKaggleスコアの違いを分析

## Notebook

実装と分析の詳細は、以下のNotebookにまとめています。

[Titanic Notebook](notebooks/titanic_machine_learning.ipynb)

## What I Learned

この取り組みを通じて、欠損値処理、カテゴリ変数の変換、
特徴量設計、モデル構築、交差検証、パラメータ調整という
表形式データ分析の基本的な流れを学びました。

また、交差検証の精度が向上してもKaggleのスコアが
必ずしも同じように向上するとは限らないことを経験し、
検証方法や特徴量の選択を慎重に行う重要性を学びました。

## Project Status

Notebookを就活用に整理しています。
