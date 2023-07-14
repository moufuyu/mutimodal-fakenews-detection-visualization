# 研究内容
人工知能学会全国大会2023でのポスター資料は```poster.pdf```に記載しています。
## 外部知識を用いたマルチモーダルフェイクニュース検出の説明性改善
フェイクニュースの多くは背景知識を含んでいるものが多い。
そのため、フェイクニュース検出そのものにおいても説明性においても、背景知識を考慮したモデルは有効であると考えられる。
入力テキストの固有表現をDBpediaで検索し、得られたIntroductionを外部知識としてモデルに入力する。モデルの判断根拠の可視化手法としてVGG19に対してはGrad-CAMを、XLNetに対してはAttentionの強さを用いる。
## フェイクニュースの可視化例
![operahouse](https://user-images.githubusercontent.com/62968285/234152107-cb9bef6a-1081-4d68-bb8f-8bfa575a2b27.png)

