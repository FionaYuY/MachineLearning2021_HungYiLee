# Self-attention 上
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
     + 怎麼決定兩個向量之間的關聯性？ 有幾種不同的做法（1)dot product（最常用）：將兩個向量分別呈上兩個不同的矩陣，得到q和k，將q和k做dot product，得到的scaler為alpha (2)additive：通過wq,wk，得到q,k，串起來丟進activation function中，再通過transform得到alpha
![self_v7 015](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/c144ee0b-a736-4ea6-be58-15427a6aecb9)
   - 計算alpha
     + a1乘上wq，得到q1。q為query
     + a2,a3,a4乘上wk，得到k。k為key
     + 將q1,k計算inner product得到alpha
     + alpha1,2表示query是1提供的，key是2提供的
     + alpha為attention score
     + ps.在實作時，q1也會跟自己算關聯性（q1,k1)，得到alpha1,1。
![self_v7 016](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/52d666c9-0c8b-4945-9c6b-d9d2c0fcc655)
   - 丟進soft-max中
     + 此soft-max與分類時的soft-max一樣（將alpha乘上exponential，在把exponential的值加起來做normalize，得到alpha prime。
     + 為什麼用soft-max？softmax最常見，但也不一定要用soft-max。用其他的也行．
![self_v7 017](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/9971b214-c110-46f5-9f8c-4d56346bb9ed)
   - 根據alpha prime，去抽取sequence中重要的資訊
     + 將a1,a2...乘上wv，得到新的向量v1,v2...
     + 將v1,v2,...乘上attention score(alpha prime)，再加起來，得到b1
     + so, 如果某一個向量他得到的attention score越高（假設a1,a2關聯性強），alpha prime很大，那經過weighted sum所得到的b1，就有可能較接近v2
     + 也就是說，誰的attention score越大，誰的v就會dominate你抽出來的結果
![self_v7 018](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/485d8f1c-5278-4d8f-9bc8-17715dc95ac3)

# self-attention 下
1. b1~b4不需要依序產生，而是以parallel的方式，同時被計算出來的
2. review:計算b2
   - a2乘上一個transform，變成q2
   - q2去對a1~a4計算attention score:
     + q2對k做dot product得到四個分數-> normalization(ex:soft-max) -> alpha prime (經過normalization後的Attention score)
   - 將alpha prime分別乘上v1~v4，全部相加，得到b2
![self_v7 020](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/c847c83b-638e-4d9c-9b06-2fb25c0c4ef3)
3. 從矩陣乘法的角度來看self-attention
   - a1~a4分別產生q,k,v
     + a乘上wq得到q
     + 
![self_v7 021](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/d3ade78c-4909-4a9c-9df4-6bc25c544598)



















