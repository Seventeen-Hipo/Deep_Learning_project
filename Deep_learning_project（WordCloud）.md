# Deep_learning_project（词云）

## 1、基础知识：

###  (1) 需要导入的包：

``` python
import matplotlib.pyplot as plt
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import numpy as np
import jieba.analyse
```

### (2) Word_Cloud参数：

~~~ python
"""
WordCloud() 可选参数
    font_path：可用于指定字体路径，包括otf和ttf
    width：词云的宽度，默认为400
    height：词云的高度，默认为200
    mask：蒙板，可用于定制词云的形状
    min_font_size：最小字号，默认为4
    max_font_size：最大字号，默认为词云的高度
    max_words：词的最大数量，默认为200
    stopwords：将被忽略的停用词，如果不指定则使用默认的停用词词库
    backgroud_color：背景颜色，默认为black
    mode：默认为RGB模式，如果为RGBA模式且background_color设为None，则背景将透明
"""
~~~

## 2、操作一（对英文文本生成词云）

``` python
# 打开英文文本
text = open('matplotlibrc.txt').read()
# 生成对象
wc = WordCloud()
wc.generate(text)
# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
# 保存词云
wc.to_file('wordcloud.png')
```

> [!NOTE]
>
> 在对英文文本生成词云操作前，需要将所选要的文本导入对应的项目文件的目录下，以便进行后续的操作

## 3、操作二（对中文文本生成词云）

``` python
# 导入所需要的包
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# 打开文件，并且设置文件编码（将文件编码设置为utf-8）
text = open('Chinese.txt',encoding='utf-8').read()
# 设置词云的相关参数，例如：生成词云的字体、宽度、高度、模式、背景颜色
wc = WordCloud(font_path='simkai.ttf',width=800, height=600,mode='RGBA',background_color=None).generate(text)
# 显示词云
plt.imshow(wc)
plt.axis("off")
plt.show()
# 保存词云
wc.to_file('ch02.png')
```

> [!IMPORTANT]
>
> 1、在导入中文文本生成词云的时候，一定要设置编码（采用 encoding = 'utf-8'），否则会报错。
>
> 1、设置生成词云的字体时，需要将相对应的字体文件导入至项目文件的目录下，此操作和编码操作同等重要。

## 4、操作三（对中文文本进行分词，并显示词云）

``` python
# 导入相关的包
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
# 打开文件
text = open('Chinese.txt',encoding='utf-8').read()
# 对中文文本进行分词，选择前100个文本文字
text = ' '.join(jieba.cut(text))
print(text[:100])
# 设置生成词语的相关参数
wc = WordCloud(font_path='simkai.ttf',width=800, height=400,mode='RGBA',background_color=None).generate(text)
# 显示词云
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()
# 保存词云
wc.to_file('ch03.png')
```

## 5、操作四（用给定的蒙版、中文文本生成词云）

``` python
# 导入相对应包
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import jieba
# 打开文件，并设置文件编码为（utf-8）
text = open('Chinese.txt',encoding='utf-8').read()
# 用jieba对中文文本进行分词
text = ' '.join(jieba.cut(text))
print(text[:100])
# 导入所需的蒙版
mask = np.array(Image.open('picture_mode.png'))
wc = WordCloud(mask=mask,font_path='simkai.ttf',mode='RGBA',background_color=None).generate(text)
# 显示词云
plt.imshow(wc)
plt.axis("off")
plt.show()
# 保存文件
wc.to_file('ch04.png')
```

> [!IMPORTANT]
>
> 要生成指定蒙版的词云的时候，需要将蒙版图片提前导入项目文件的目录下。

## 6、操作五（将生成的词云，设置文字颜色）

``` python
# 导入相对应的包
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import random # 用随机函数设置词云文字的颜色
import numpy as np
import jieba

# 打开文本
text = open('Chinese.txt',encoding='utf-8').read()

# 中文分词
text = ' '.join(jieba.cut(text))
print(text[:100])

# 颜色函数
def random_color(word,font_size,position,orientation,font_path,random_state):
    s = 'hsl(0,%d%%,%d%%)' % (random.randint(60,80),random.randint(60,80))
    print(s)
    return s

# 生成对象
mask = np.array(Image.open('picture_mode.png'))
wc = WordCloud(color_func=random_color,mask=mask,font_path='simkai.ttf',mode='RGBA',background_color=None).generate(text)

plt.imshow(wc)
plt.axis("off")
plt.show()

wc.to_file('ch05.png')
```

> [!NOTE]
>
> 在上述代码的颜色函数中，使用的时HSL的配色方案。其他的HSL的配色方案，可以参考：https://www.w3.org/wiki/CSS3/Color/HSL

## 7、操作六（精细的控制词云中出现的词，以及每个词的大小）

``` python
"""
如果希望精细的控制词云中出现的词，以及每个词的大小，可以尝试generate_from_frequencies()
frequencise:一个字典，用于指词和对应的大小
max_font_size:最大字号，默认为None

generate() = process() + generate_from_frequencies()
"""
import numpy as np
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import jieba.analyse

# 打开文本
text = open('Chinese.txt',encoding='utf-8').read()

# 提取关键字
freq = jieba.analyse.extract_tags(text,topK=100,withWeight=True) # 是一个列表
# topK 表示所需要提取关键字的数目
# withWeight 表示输出所 提取关键词的权重
print(freq[:20])
freq = {i[0]:i[1] for i in freq} # 将其转换为字典

# 生成对象
mask = np.array(Image.open('picture_mode.png'))
wc = WordCloud(mask=mask,font_path='simkai.ttf',mode='RGBA',background_color=None).generate_from_frequencies(freq)

# 从图片中生成颜色
image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

# 显示词云
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存
wc.to_file('ch06.png')
```

