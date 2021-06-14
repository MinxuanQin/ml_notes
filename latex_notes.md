## Latex 入门笔记

### documentclass

控制文件的appearance，最常见的是article。

### content

由`\begin{document}`, `\end{document}` tag构成。代表文件的主体（body）

### preamble 前言

在Latex中，所有在`\begin{document}`之前插入的叫做**preamble**。如：文件类型`\documentclass`,`usepackage`

```
\documentclass[12pt, letterpaper]{article}
\usepackage[utf8]{inputenc}
```

字体大小默认为10pt，paper还可以选a4paper, legalpaper

推荐使用utf-8作为文档的编码格式

### title, author, date

在preamble部分添加。

```
\title{First document}
\author{Hubert Farnsworth}
\thanks{funded by Overleaf team}
\date{February 2014}
```
在正文添加`\maketitle`来生成标题。

### comments

在行首加%即可

### 测试发现

1. `\maketitle`指令的位置决定了在何处插入标题

2. `\today`不会只取代标题下日期的位置， 还会额外插入一页（thanks同理）
**目前没掌握原因**

3. `\thanks`也没有起到致谢的作用，不能放在前言处，目前看来是注释的作用。 会产生奇怪的第一页，还是先别用了