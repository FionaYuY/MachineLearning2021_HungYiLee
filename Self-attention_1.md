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
     + each vector has a label(ex:input為4個向量，則輸出也是4個向量）。
       * ex: pos tagging(詞性標注）
       * ex: 語音（每一個vector都是一個phone)
       * ex: social network（每一個節點會不會買一個商品？）
![self_v7 007](https://github.com/FionaYuY/MachineLearning2021_HungYiLee/assets/151610467/68573c66-40e6-40e2-b2e5-f283a27d761e)













