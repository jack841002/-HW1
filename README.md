

```python
資料集：
台電2017/01/01到2019/03/31
ps:4/1的data要等4/2才會有，所以我4/1的data是用台電預測4/1號的值

預測方法：
feature只用"日期", "尖峰負載(MW)", "備轉容量(MW)", "備轉容量率(%)"
然後利用前7天預測未來7天，分割資料的部分我是用9:1當作train和val
model是用LSTM來實作，得出的model結果再跟前2年的春假做平均得到最後的預測值
再把結果存到D:\jupyter


nbviewer產生固定連結：
https://nbviewer.jupyter.org/gist/jack841002/0c52239415858fdf5a73dde6b487c631
```
