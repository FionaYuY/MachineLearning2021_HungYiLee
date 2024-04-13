# Self-attention
1. self-attention想要解決的問題
   - 以往的input都是一個向量，輸出則可能是scalar（regression）或是class（classification)
   - 利用self-attention則可使用a set of vectors當作input，並且長度是可改變的、不一樣的
2. vector set as input
   - ex: 文字處理、聲音訊號
3. 怎麼把詞彙表示為向量
![self_v7 003](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/c6c0064f-46e9-438c-8492-a503a2351a5e)
   - One-hot encoding
     + 每一個向量的維度跟所有存在的字彙的數量相同
     + 缺點：假設所有詞彙彼此都是沒有關係的，看不出字彙之間的關聯
   - Word Embedding
     + https://www.youtube.com/watch?v=X7PH3NuYW0Q
     + 給每一個字彙一個向量，並且每一個向量是有語意資訊的（有關連的字彙會集結成一團）
     + 一個句子就是一排長度不一的向量
4. 聲音訊號也是vector set as input
![self_v7 004](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/4eead2f5-e3a6-4716-b8fd-ae888791c92f)
   - 將一段聲音訊號取一段範圍（一個window)（可能是25ms)，將其資訊描述成一個向量，此向量稱為一個frame
   - 有很多方法可以將一段聲音訊號轉換為向量
   - 將訊號範圍往右移10ms（通常），以此類推
   - 1s有100 frames
5. graph is also a set of vectors
   - ex: social network就是一個graph，上面的每一個節點（人）都是一個向量
![self_v7 005](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/1b03ab9f-6ab6-40b6-8e8e-0605a9e569f0)
   - ex: 分子也可以看做graph。每一個原子都是一個向量，可以用one-hot encoding表示原子
![self_v7 006](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/dd66c04c-88e7-4f4a-8b03-4f8bed5d951b)
6. what is the output
   - 三種可能性
     + each vector has a label(ex:input為4個向量，則輸出也是4個向量）。(輸入跟輸出一樣多）-> 又稱sequence labeling
       * ex: pos tagging(詞性標注）
       * ex: 語音（每一個vector都是一個phone)
       * ex: social network（每一個節點會不會買一個商品？）
![self_v7 007](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/68573c66-40e6-40e2-b2e5-f283a27d761e)
     + the whole sequence has a label
       * ex: sentiment analysis
       * ex: 聽一段聲音決定是誰講的
       * ex: 給一個分子，預測其毒性or親水性...
![self_v7 008](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/6ee4c622-b5be-4696-9116-2284dea4894c)
     + model decides the number of labels itself -> 又稱seq2seq
       * ex: tranlsation
       * 語音辨識（輸入一句話，輸出一段文字）
7. how to solve sequence labeling?
   - 直覺的想法是用fully-connected的network，各個擊破
     + 瑕疵：ex: I saw a saw(鋸子），對於fully-connected network而言，兩個saw的意義一樣，輸出也會是同一個東西
   - is it possible to consider the context?
     + 將每一個向量的前後幾個向量串起來，一起丟進fully-connected network。
     + 給fully-connected 整個window的資訊，讓其可以考慮上下文（前後幾個向量）
     + 瑕疵：沒辦法考慮整個sequence。若開一個可以覆蓋整篇sequence的window->太長了，運算量大，且有可能overfitting
![self_v7 010](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/dc85b63c-fed2-4ea2-aa2d-fa14e74ad9c8)
   - self-attention !!!
8. self-attention
   - 會吃進一整個sequence的資訊
   - input幾個vector，則輸出幾個vector
   - 輸出的vector是考慮一整個sequence才得到的
   - 再把輸出的vector丟進fully-connected network，得到的結果便是考慮了整個sequence所得到的結果
 ![self_v7 011](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/2743b00d-b1cf-4d05-927a-60a8f1be9818)
   - 可以不只用一次，可以疊加。也可以將self-attention和fully-connected交替使用
![self_v7 012](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/fa932ae9-579e-42d8-91d8-9e26eb94c4fd)
   - 最重要的應用:transformer
9. self-attention的運作
![self_v7 013](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/3348087a-ba5a-4b8c-a783-84d2ebbf92dc)
   - input是一串vector，可能是整個network的input，也可能是某個hidden layer的output
   - 怎麼產生b1這個向量（剩下的以此類推）
     + 根據a1，找出sequence中與a1相關的其他向量（find the relevant vectors in a sequence)
     + 每一個向量跟a1的關聯程度以alpha來表示
![self_v7 014](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/099fd755-bb19-4037-b88c-6abf52b8be86)
     + 怎麼決定兩個向量之間的關聯性？ 有幾種不同的做法（1)dot product：將兩個向量分別呈上兩個不同的矩陣，得到q和k，將q和k做dot product，得到的scaler為alpha (2)additive：通過wq,wk
![self_v7 015](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/c144ee0b-a736-4ea6-be58-15427a6aecb9)






















